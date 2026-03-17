FROM node:18-slim

# Installa Python 3.11, nvidia-smi e dipendenze necessarie
# Includi build-essential e ninja-build per compilazione CUDA JIT (flashinfer)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Installa nvidia-smi (se disponibile)
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | apt-key add - || true

WORKDIR /app

# Copia package.json e installa dipendenze
COPY package.json .
RUN npm install

# Copia il codice
COPY server.js .

# Esponi la porta dell'API backend (3001) e la porta vLLM OpenAI API (8000)
EXPOSE 3001 8000

# Avvia il server
CMD ["npm", "start"]
