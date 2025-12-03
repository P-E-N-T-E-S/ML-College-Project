# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install JupyterLab and data science packages
RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotly \
    ipywidgets

# Create workspace directory
RUN mkdir -p /workspace

# Expose JupyterLab port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    JUPYTER_ENABLE_LAB=yes

# Start JupyterLab without authentication for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
