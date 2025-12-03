# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories for MLflow artifacts and metrics
RUN mkdir -p /app/mlruns /app/metrics

# Make the startup script executable
RUN chmod +x start_mlflow_ui.sh

# Expose MLflow UI port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Use the startup script as entrypoint
CMD ["./start_mlflow_ui.sh"]
