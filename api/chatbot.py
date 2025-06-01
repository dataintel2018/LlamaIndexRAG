import os
from typing import List, Optional
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatbotInterface, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            print("Initializing RAG Chatbot...")
            try:
                self.documents = self._load_documents()
                print("Documents loaded successfully!")
            except ValueError as e:
                print(f"Warning: {e}")
                print("Continuing without documents...")
                self.documents = None
            
            print("Initializing components...")
            storage_context, service_context = self._initialize_rag_components()
            
            print("Creating index...")
            self.index = self._create_or_load_index(self.documents, storage_context, service_context)
            self.query_engine = self.index.as_query_engine()
            
            self._is_initialized = True

    def _load_documents(self, data_dir: str = "data") -> Optional[List]:
        """Load documents from the data directory."""
        try:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                return None
            
            reader = SimpleDirectoryReader(data_dir)
            documents = reader.load_data()
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            return None

    def _initialize_rag_components(self):
        """Initialize RAG components including LLM, embeddings, and vector store."""
        try:
            # Initialize Ollama with host from environment variable
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            llm = Ollama(model="llama3.2", temperature=0.7, base_url=ollama_host)
            print(f"Connecting to Ollama at: {ollama_host}")
            
            # Initialize embeddings
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
            
            # Initialize ChromaDB with a persistent directory
            db_dir = os.path.join(os.getcwd(), "chroma_db")
            os.makedirs(db_dir, exist_ok=True)
            
            db = chromadb.PersistentClient(path=db_dir)
            chroma_collection = db.get_or_create_collection("rag_collection")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create service context
            service_context = ServiceContext.from_defaults(
                llm=llm,
                embed_model=embed_model,
            )
            
            return storage_context, service_context
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise

    def _create_or_load_index(self, documents, storage_context, service_context):
        """Create or load the vector store index."""
        try:
            if documents:
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    service_context=service_context
                )
            else:
                index = VectorStoreIndex.from_vector_store(
                    storage_context.vector_store,
                    service_context=service_context
                )
            return index
        except Exception as e:
            print(f"Error creating/loading index: {e}")
            raise

    def process_query(self, message: str) -> str:
        """Process a query and return the response."""
        if not message.strip():
            return "Please provide a non-empty message."
        
        try:
            response = self.query_engine.query(message)
            return str(response.response)
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(error_message)
            return error_message 