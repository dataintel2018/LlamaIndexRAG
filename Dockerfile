FROM ollama/ollama:latest

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

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
CMD ["/usr/bin/supervisord"] 