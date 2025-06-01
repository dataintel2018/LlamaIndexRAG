from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from chatbot import ChatbotInterface

# Initialize FastAPI
app = FastAPI()

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

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}

def create_gradio_interface():
    """Create and configure the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown("# RAG Chatbot with LlamaIndex and Ollama")
        gr.Markdown("Ask questions about your documents using this RAG-powered chatbot.")
        
        chatbot_ui = gr.Chatbot(
            height=500,
            show_copy_button=True,
            render_markdown=True
        )
        
        with gr.Row():
            with gr.Column(scale=8):
                msg = gr.Textbox(
                    label="Type your message here...",
                    placeholder="Type your question and press Enter or click Submit",
                    lines=2
                )
            with gr.Column(scale=2):
                submit = gr.Button("Submit", variant="primary")
        
        with gr.Row():
            clear = gr.Button("Clear Chat")
        
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
        
        # Connect the components
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
        
        with gr.Accordion("Example Questions", open=False):
            gr.Examples(
                examples=[
                    "What are the main topics in the documents?",
                    "Can you summarize the key points?",
                    "What are the most important findings?",
                ],
                inputs=msg
            )
    
    return iface

# Create the Gradio interface
gradio_app = create_gradio_interface()

# Mount the Gradio app to FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/")

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 