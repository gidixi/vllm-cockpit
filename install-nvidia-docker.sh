#!/bin/bash

# Script per installare nvidia-docker (nvidia-container-toolkit)
# Esegui: sudo ./install-nvidia-docker.sh

set -euo pipefail

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verifica che lo script sia eseguito come root
if [[ $EUID -ne 0 ]]; then
    log_error "Questo script deve essere eseguito come root o con sudo"
    log_info "Esegui: sudo ./install-nvidia-docker.sh"
    exit 1
fi

log_info "Installazione nvidia-container-toolkit per Docker"
echo ""

# Verifica che Docker sia installato
if ! command -v docker &> /dev/null; then
    log_error "Docker non trovato!"
    log_info "Installa Docker prima di continuare"
    exit 1
fi

log_info "✅ Docker trovato: $(docker --version)"
echo ""

# Verifica che nvidia-smi funzioni
if ! command -v nvidia-smi &> /dev/null; then
    log_warn "nvidia-smi non trovato. La GPU potrebbe non essere disponibile."
else
    log_info "✅ NVIDIA driver trovato"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    echo ""
fi

# Aggiungi il repository NVIDIA
log_info "Aggiunta repository NVIDIA..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Aggiorna repository
log_info "Aggiornamento repository..."
apt-get update

# Installa nvidia-container-toolkit
log_info "Installazione nvidia-container-toolkit..."
apt-get install -y nvidia-container-toolkit

# Configura Docker per usare il runtime NVIDIA
log_info "Configurazione Docker runtime..."
nvidia-ctk runtime configure --runtime=docker

# Riavvia Docker
log_info "Riavvio Docker..."
systemctl restart docker

# Verifica installazione
log_info "Verifica installazione..."
if docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    log_info "✅ nvidia-docker installato correttamente!"
    echo ""
    log_info "Ora puoi decommentare la sezione GPU in docker-compose.yml"
else
    log_warn "⚠️  Installazione completata, ma verifica fallita"
    log_info "Prova manualmente: docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi"
fi

echo ""
log_info "✅ Installazione completata!"
log_info ""
log_info "Per abilitare la GPU in docker-compose.yml:"
log_info "  1. Apri docker-compose.yml"
log_info "  2. Decommenta le righe 25-31 (sezione deploy con GPU)"
log_info "  3. Riavvia: docker compose down && docker compose up -d"
