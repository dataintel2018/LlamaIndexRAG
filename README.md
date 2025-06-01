# RAG Chatbot with LlamaIndex and Ollama

This project implements a Retrieval-Augmented Generation (RAG) chatbot using LlamaIndex, Ollama, and ChromaDB for vector storage.

## Prerequisites

- Python 3.9+
- Ollama installed and running locally
- A compatible LLM model downloaded in Ollama (e.g., llama2)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Ollama is running with your preferred model (e.g., llama2):
```bash
ollama run llama2
```

3. Create a `data` directory and add your documents for the knowledge base.

## Usage

1. Place your documents in the `data` directory
2. Run the chatbot:
```bash
python chatbot.py
```

3. Start chatting with the bot! The bot will use your documents as context to provide more accurate and relevant responses.

## Project Structure

- `chatbot.py`: Main chatbot implementation
- `data/`: Directory for storing documents
- `requirements.txt`: Python dependencies 