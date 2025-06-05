import os
import shutil
import datetime
import pandas as pd
from typing import List, Optional, Callable
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document
from llama_index.llms import Ollama
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb
from pydantic import BaseModel

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None

class ChatbotInterface:
    _instance = None
    _is_initialized = False
    MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB in bytes

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatbotInterface, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            print("Initializing RAG Chatbot...")
            self._initialize_components()
            self._is_initialized = True

    def _initialize_components(self):
        """Initialize all components of the chatbot."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Initialize Ollama with host from environment variable
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.llm = Ollama(model="llama3.2", temperature=0.7, base_url=ollama_host)
            print(f"Connecting to Ollama at: {ollama_host}")
            
            # Initialize embeddings
            #self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
            self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")            
            
            # Initialize ChromaDB with a persistent directory
            db_dir = os.path.join(os.getcwd(), "chroma_db")
            os.makedirs(db_dir, exist_ok=True)
            
            self.db = chromadb.PersistentClient(path=db_dir)
            self.chroma_collection = self.db.get_or_create_collection("rag_collection")
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Create service context
            self.service_context = ServiceContext.from_defaults(
                llm=self.llm,
                embed_model=self.embed_model,
            )
            
            # Initialize empty index
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                service_context=self.service_context
            )
            self.query_engine = self.index.as_query_engine()
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise

    def _get_documents(self, data_dir: str = "data") -> Optional[List]:
        """Load documents from the data directory."""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            return None
        
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if not files:  # If directory is empty
            return None
            
        reader = SimpleDirectoryReader(data_dir)
        return reader.load_data()

    def _load_documents(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Load documents and create/update the index."""
        try:
            if progress_callback:
                progress_callback("Loading documents...")
            
            documents = self._get_documents()
            if documents:
                if progress_callback:
                    progress_callback(f"Creating embeddings for {len(documents)} documents...")
                
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=self.storage_context,
                    service_context=self.service_context
                )
                self.query_engine = self.index.as_query_engine()
                
                if progress_callback:
                    progress_callback("Document processing complete!")
            else:
                if progress_callback:
                    progress_callback("No documents found to process.")
                
        except Exception as e:
            error_msg = f"Error loading documents: {e}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            raise

    def get_document_count(self) -> int:
        """Get the number of documents in the data directory."""
        data_dir = "data"
        if not os.path.exists(data_dir):
            return 0
        return len([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])

    def create_embeddings(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Create embeddings for all documents in the data directory."""
        try:
            if progress_callback:
                progress_callback("Starting embedding creation...")
            
            doc_count = self.get_document_count()
            if doc_count == 0:
                if progress_callback:
                    progress_callback("No documents found to process.")
                return
            
            if progress_callback:
                progress_callback(f"Creating embeddings for {doc_count} documents...")
            
            self._load_documents(progress_callback)
            #self.load_csv("data/test.csv", ["text"], ["metadata"], progress_callback)
            
            if progress_callback:
                progress_callback("Embedding creation complete!")
        except Exception as e:
            error_msg = f"Error creating embeddings: {e}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            raise

    def process_query(self, message: str) -> str:
        """Process a query and return the response."""
        if not message.strip():
            return "Please provide a non-empty message."
        
        if self.get_document_count() == 0:
            return "No documents have been uploaded yet. Please upload some documents first."
        
        try:
            response = self.query_engine.query(message)
            return str(response.response)
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(error_message)
            return error_message

    def validate_file_size(self, file) -> bool:
        """Validate that file size is within limits."""
        try:
            # For FastAPI UploadFile objects
            if hasattr(file, "file"):
                file.file.seek(0, 2)  # Seek to end of file
                size = file.file.tell()
                file.file.seek(0)  # Reset file pointer
                return size <= self.MAX_FILE_SIZE
            # For Gradio file upload
            elif hasattr(file, "name"):
                return os.path.getsize(file.name) <= self.MAX_FILE_SIZE
            return False
        except Exception:
            return False

    def upload_files(self, files, progress_callback=None) -> tuple[list, list]:
        """
        Upload files to the data directory. Handles both regular documents and CSV files.
        
        Args:
            files: List of file objects (either FastAPI UploadFile or Gradio file objects)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            tuple: (successful_uploads, failed_uploads)
        """
        if progress_callback:
            progress_callback("Starting file upload...")
        
        successful_uploads = []
        failed_uploads = []
        
        for file in files:
            try:
                # Get filename based on file type
                if hasattr(file, "file"):  # FastAPI UploadFile
                    filename = file.filename
                    is_fastapi = True
                elif hasattr(file, "name"):  # Gradio file upload
                    filename = os.path.basename(file.name)
                    is_fastapi = False
                else:
                    continue

                # Check file size
                if not self.validate_file_size(file):
                    failed_uploads.append((filename, "Exceeds size limit"))
                    continue

                # Handle duplicate filenames
                dest_path = os.path.join("data", filename)
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{name}_{timestamp}{ext}"
                    dest_path = os.path.join("data", filename)

                # Save the file
                if is_fastapi:
                    with open(dest_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                else:
                    shutil.copy2(file.name, dest_path)

                successful_uploads.append(filename)
                
                if progress_callback:
                    progress_callback(f"Uploaded: {filename}")
                    
            except Exception as e:
                error_msg = str(e)
                failed_uploads.append((
                    getattr(file, 'filename', getattr(file, 'name', 'Unknown file')),
                    error_msg
                ))
                if progress_callback:
                    progress_callback(f"Failed to upload: {error_msg}")
        
        if progress_callback:
            progress_callback("Upload process complete!")
        
        return successful_uploads, failed_uploads

    def load_csv(self, csv_path: str, text_columns: List[str], metadata_columns: Optional[List[str]] = None, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Load a CSV file into the vector store.
        
        Args:
            csv_path: Path to the CSV file
            text_columns: List of column names whose content should be combined into the document text
            metadata_columns: Optional list of column names to include as metadata
            progress_callback: Optional callback function for progress updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("mahesh1")
            if progress_callback:
                progress_callback("Loading CSV file...")
            
            print("mahesh2")
            print(csv_path)
            # Read the CSV file
            from read_csv import read_csv_file
            df = read_csv_file(csv_path)
            #df = pd.read_csv(csv_path)
            print("mahesh5")
            
            if progress_callback:
                progress_callback(f"Processing {len(df)} rows...")
            
            # Create documents from the CSV rows
            documents = []
            for idx, row in df.iterrows():
                print("mahesh6")
                # Combine specified text columns
                text_columns = df.columns.tolist()
                text_content = " ".join(str(row[col]) for col in text_columns if pd.notna(row[col]))
                print("mahesh7")
                # Create metadata dictionary if metadata columns are specified
                metadata = {}
                print("mahesh8")
                if metadata_columns:
                    metadata = {col: str(row[col]) for col in metadata_columns if pd.notna(row[col])}
                print("mahesh9")
                # Add row index as metadata
                metadata['row_index'] = idx
                
                # Create document
                print("mahesh3")
                doc = Document(text=text_content, metadata=metadata)
                documents.append(doc)
                print("mahesh4")
            
            if progress_callback:
                progress_callback("Creating embeddings...")
            
            # Add documents to the index
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                service_context=self.service_context
            )
            
            # Update query engine
            self.query_engine = self.index.as_query_engine()
            
            if progress_callback:
                progress_callback("CSV processing complete!")
            
            return True
            
        except Exception as e:
            raise
            error_msg = f"Error loading CSV file: {str(e)}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            return False 