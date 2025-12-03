FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar requirements do streaming
COPY stream/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY stream/stream_simulator.py /app/stream_simulator.py
COPY stream/thingsboard_client.py /app/thingsboard_client.py
COPY stream/.env /app/.env

# Verificar estrutura (debug)
RUN echo "=== Estrutura de diretórios ===" && \
    ls -la /app/ && \
    echo "=== Conteúdo de /app/src/ ===" && \
    ls -la /app/src/ || echo "src/ não encontrado"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/src

# Comando padrão
CMD ["python", "stream_simulator.py", "--delay", "2"]