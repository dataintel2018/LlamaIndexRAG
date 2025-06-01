#!/bin/bash

# Make the script executable
chmod +x setup.sh

# Create necessary directories
mkdir -p data chroma_db

# Build and start the containers
docker-compose up -d

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
sleep 10

# Pull the llama3.2 model
docker exec ollama ollama pull llama3.2

echo "Setup complete! The application is running at http://localhost:8000"
echo "You can now add your documents to the 'data' directory." 