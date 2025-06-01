from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from chatbot import ChatbotInterface
import os
import shutil
from typing import List
from fastapi.responses import JSONResponse
import datetime

print("="*50)
print("Loading index.py module")
print("="*50)

# Initialize FastAPI
app = FastAPI()

# Constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB in bytes

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chatbot
chatbot = ChatbotInterface()

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def validate_file_size(file):
    """Validate that file size is within limits."""
    # Get file size - for UploadFile objects
    if hasattr(file, "file"):
        file.file.seek(0, 2)  # Seek to end of file
        size = file.file.tell()
        file.file.seek(0)  # Reset file pointer
        return size <= MAX_FILE_SIZE
    # For Gradio file upload
    elif isinstance(file, str):
        return os.path.getsize(file) <= MAX_FILE_SIZE
    return False

# @app.post("/upload")
# async def upload_files(files: List[UploadFile] = File(...)):
#     """
#     Upload one or more files to be used for RAG.
#     Files will be saved in the data directory.
#     """
#     try:
#         successful_uploads, failed_uploads = chatbot.upload_files(files)
        
#         if failed_uploads:
#             # If some files failed but others succeeded
#             if successful_uploads:
#                 return JSONResponse(
#                     content={
#                         "message": "Some files uploaded successfully, others failed",
#                         "successful_uploads": successful_uploads,
#                         "failed_uploads": [f"{name} ({reason})" for name, reason in failed_uploads]
#                     },
#                     status_code=207  # Multi-Status
#                 )
#             # If all files failed
#             else:
#                 return JSONResponse(
#                     content={
#                         "error": "All files failed to upload",
#                         "failed_uploads": [f"{name} ({reason})" for name, reason in failed_uploads]
#                     },
#                     status_code=400
#                 )
        
#         # All files succeeded
#         return JSONResponse(
#             content={
#                 "message": "Files uploaded successfully",
#                 "uploaded_files": successful_uploads
#             },
#             status_code=200
#         )
#     except Exception as e:
#         return JSONResponse(
#             content={"error": str(e)},
#             status_code=500
#         )

@app.post("/create-embeddings")
async def create_embeddings():
    """Create embeddings for all uploaded documents."""
    try:
        doc_count = chatbot.get_document_count()
        if doc_count == 0:
            return JSONResponse(
                content={"message": "No documents found to process."},
                status_code=400
            )
        
        chatbot.create_embeddings()
        return JSONResponse(
            content={
                "message": "Embeddings created successfully",
                "processed_documents": doc_count
            },
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/documents")
async def list_documents():
    """List all documents currently available for RAG."""
    try:
        documents = os.listdir("data")
        return {"documents": documents}
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


def handle_file_upload(files):
    """Handle file upload from Gradio interface."""
    if not files:
        return "Please select files to upload."
    
    try:
        status_messages = []
        
        def update_progress(msg):
            status_messages.append(msg)
        
        successful_uploads, failed_uploads = chatbot.upload_files(files, update_progress)
        
        # Prepare final status message
        result_messages = []
        if successful_uploads:
            result_messages.append(f"✅ Successfully uploaded {len(successful_uploads)} files: {', '.join(successful_uploads)}")
        if failed_uploads:
            result_messages.append(f"❌ Failed to upload {len(failed_uploads)} files: {', '.join(f'{name} ({reason})' for name, reason in failed_uploads)}")
        
        if not result_messages:
            return "No files were processed."
        
        return "\n".join(result_messages)
        
    except Exception as e:
        error_msg = f"❌ Error during upload process: {str(e)}"
        print(error_msg)  # Log the error
        return error_msg

def handle_embedding_creation(progress=gr.Progress()):
    """Handle embedding creation from Gradio interface."""
    try:
        doc_count = chatbot.get_document_count()
        if doc_count == 0:
            return "No documents found to process."
        
        progress(0, desc="Starting embedding creation...")
        
        def update_progress(msg):
            if "Creating embeddings for" in msg:
                progress(0.3, desc=msg)
            elif "Document processing complete" in msg:
                progress(1.0, desc="Embeddings created successfully!")
            else:
                progress(0.6, desc=msg)
            return msg
        
        chatbot.create_embeddings(update_progress)
        return f"Successfully created embeddings for {doc_count} documents."
    except Exception as e:
        error_msg = f"Error creating embeddings: {str(e)}"
        print(error_msg)
        return error_msg

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    with gr.Blocks(css="""
        button.gr-button {
            background-color: #1a4b84 !important;
            border: none !important;
            color: white !important;
        }
        button.gr-button:hover {
            background-color: #0d3868 !important;
            color: white !important;
        }
        .gr-button.gr-button-lg {
            background-color: #1a4b84 !important;
        }
        .gr-button.gr-button-lg:hover {
            background-color: #0d3868 !important;
        }
        div.gr-button {
            background-color: #1a4b84 !important;
        }
        div.gr-button:hover {
            background-color: #0d3868 !important;
        }
    """) as iface:
        gr.Markdown(
            """
            # RAG Chatbot with LlamaIndex
            
            Ask questions about your documents using this RAG-powered chatbot. Upload your documents and let AI help you analyze them.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # File Upload Section
                with gr.Group():
                    gr.Markdown("### Document Management")
                    upload_button = gr.UploadButton(
                        "Upload Files",
                        file_count="multiple",
                        file_types=[".txt", ".pdf", ".doc", ".docx", ".md"]
                    )
                    status_output = gr.Textbox(
                        label="Upload Status",
                        placeholder="Upload status will appear here...",
                        interactive=False,
                        lines=3
                    )
                    gr.Markdown("*Supported: PDF, TXT, DOC, DOCX, MD files (Max 100MB)*")
                    create_embeddings_button = gr.Button("Create Embeddings")
                    embedding_status = gr.Textbox(
                        label="Processing Status",
                        placeholder="Processing status will appear here...",
                        interactive=False,
                        lines=2
                    )

            with gr.Column(scale=3):
                # Chat Interface
                with gr.Group():
                    gr.Markdown("### Chat Interface")
                    chatbot_ui = gr.Chatbot(
                        height=400,
                        show_copy_button=True,
                        render_markdown=True,
                        container=True
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about your documents...",
                            lines=2,
                            show_label=False
                        )
                        submit = gr.Button("Send")
                    
                    with gr.Row():
                        clear = gr.Button("Clear Chat")
                        
                    with gr.Accordion("Example Questions", open=False):
                        gr.Examples(
                            examples=[
                                "What are the main topics discussed in these documents?",
                                "Can you provide a summary of the key points?",
                                "What are the most important findings or conclusions?",
                                "How do different documents relate to each other?",
                                "What are the recommendations or next steps mentioned?"
                            ],
                            inputs=msg
                        )
        
        # Connect the components
        def user_input(user_message, history):
            if not user_message:
                return "", history
            try:
                bot_response = chatbot.process_query(user_message)
                history = history or []
                history.append((user_message, bot_response))
                return "", history
            except Exception as e:
                error_message = f"Error: {str(e)}"
                history = history or []
                history.append((user_message, error_message))
                return "", history

        upload_button.upload(
            fn=handle_file_upload,
            inputs=upload_button,
            outputs=status_output,
            show_progress=True
        )
        
        create_embeddings_button.click(
            fn=handle_embedding_creation,
            inputs=None,
            outputs=embedding_status
        )
        
        submit.click(
            user_input,
            inputs=[msg, chatbot_ui],
            outputs=[msg, chatbot_ui],
        )
        
        msg.submit(
            user_input,
            inputs=[msg, chatbot_ui],
            outputs=[msg, chatbot_ui],
        )
        
        clear.click(lambda: None, None, chatbot_ui, queue=False)
    
    return iface

# Create the Gradio interface
gradio_app = create_gradio_interface()

# Mount the Gradio app to FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/")

# For local development
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 