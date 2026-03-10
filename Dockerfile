FROM python:3.11-slim

# Installa dipendenze di sistema necessarie per vLLM
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crea directory di lavoro
WORKDIR /app

# Il venv verrà montato come volume, quindi non lo creiamo qui
# Copiamo solo i file necessari
COPY requirements.txt .

# Esponi la porta del server
EXPOSE 8000

# Il venv viene montato come volume, quindi usiamo quello
# Comando di default (può essere sovrascritto)
CMD ["/app/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server", "--help"]
