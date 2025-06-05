FROM python:3.9-slim
#FROM ollama/ollama:latest

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install curl if it's not already present
RUN apt-get update && apt-get install -y curl

RUN curl -fsSL https://ollama.com/install.sh | sh
RUN ollama pull llama3.2
RUN ollama serve

# Copy your application
COPY . /app
WORKDIR /app

# Install your application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Setup supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create necessary directories
RUN mkdir -p /app/data /app/chroma_db

# Set environment variable for local connection
ENV OLLAMA_HOST=http://localhost:11434

# Expose ports
EXPOSE 11434 8000

# Start supervisord
# Command to run the application
CMD ["python", "api/index.py"] 