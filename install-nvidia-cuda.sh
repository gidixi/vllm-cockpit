#!/bin/bash

# Script di installazione driver NVIDIA con supporto CUDA
# Supporta: Quadro P2000, RTX 3090, RTX 4090 e altre schede NVIDIA
#
# Utilizzo:
#   sudo ./install-nvidia-cuda.sh

set -euo pipefail

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzioni per log
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_question() {
    echo -e "${BLUE}[?]${NC} $1"
}

# Verifica che lo script sia eseguito come root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "Questo script deve essere eseguito come root o con sudo"
        exit 1
    fi
    log_info "Verifica permessi: OK"
}

# Verifica presenza scheda NVIDIA
check_nvidia_gpu() {
    log_info "Verifica presenza scheda NVIDIA..."
    
    if ! lspci | grep -i "nvidia\|vga" | grep -i nvidia > /dev/null; then
        log_error "Nessuna scheda NVIDIA rilevata!"
        log_error "Esegui: lspci | grep -i vga"
        exit 1
    fi
    
    local gpu_info=$(lspci | grep -i "nvidia\|vga" | grep -i nvidia)
    log_info "Scheda NVIDIA rilevata:"
    echo "  $gpu_info"
    
    # Verifica se è già installato nvidia-smi e se funziona
    if command -v nvidia-smi &> /dev/null; then
        log_info "nvidia-smi trovato, verifica funzionamento driver..."
        if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader &> /dev/null; then
            log_warn "Driver NVIDIA già installato e funzionante:"
            nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
            read -p "Vuoi continuare comunque? (s/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Ss]$ ]]; then
                log_info "Installazione annullata"
                exit 0
            fi
        else
            log_warn "nvidia-smi trovato ma driver non funzionante (probabilmente non installato o non caricato)"
            log_info "Procedo con l'installazione del driver..."
        fi
    fi
}

# Abilita repository non-free e contrib
enable_nonfree_repos() {
    log_info "Abilitazione repository non-free e contrib..."
    
    # Rimuovi repository CUDA esistenti che potrebbero causare problemi (verranno riaggiunti dopo)
    if [ -f /etc/apt/sources.list.d/cuda.list ]; then
        log_info "Rimozione repository CUDA esistente (verrà riaggiunto dopo)..."
        rm -f /etc/apt/sources.list.d/cuda.list
    fi
    
    local sources_file="/etc/apt/sources.list"
    local backup_file="${sources_file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Crea backup
    cp "$sources_file" "$backup_file"
    log_info "Backup creato: $backup_file"
    
    # Rileva codename Debian/Ubuntu
    local codename=""
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        codename="$VERSION_CODENAME"
    else
        log_error "Impossibile rilevare il codename della distribuzione"
        exit 1
    fi
    
    # Abilita non-free e contrib
    if ! grep -q "contrib non-free" "$sources_file"; then
        sed -i 's/main non-free-firmware/main non-free-firmware contrib non-free/g' "$sources_file"
        sed -i "s/${codename}-security main/${codename}-security main contrib non-free/g" "$sources_file"
        sed -i "s/${codename}-updates main/${codename}-updates main contrib non-free/g" "$sources_file"
        log_info "Repository non-free e contrib abilitati"
    else
        log_warn "Repository non-free e contrib già abilitati"
    fi
    
    # Aggiorna repository
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    log_info "Repository aggiornati"
}

# Installa dipendenze base
install_dependencies() {
    log_info "Installazione dipendenze base..."
    
    # Verifica che non ci siano altri processi apt in esecuzione
    local max_wait=300  # 5 minuti
    local waited=0
    while pgrep -x apt-get > /dev/null || pgrep -x dpkg > /dev/null; do
        if [ $waited -ge $max_wait ]; then
            log_error "Altri processi apt/dpkg in esecuzione da troppo tempo"
            log_error "Attendi che finiscano o termina manualmente i processi"
            exit 1
        fi
        log_warn "Attendo che altri processi apt/dpkg finiscano... ($waited/$max_wait secondi)"
        sleep 5
        waited=$((waited + 5))
    done
    
    # Verifica kernel headers
    local kernel_version=$(uname -r)
    local headers_package="linux-headers-${kernel_version}"
    
    log_info "Kernel version: $kernel_version"
    log_info "Installazione: build-essential, dkms, $headers_package..."
    
    # Mostra output per debug (rimosso > /dev/null 2>&1)
    if ! apt-get install -y --no-install-recommends \
        build-essential \
        dkms \
        "$headers_package" \
        curl \
        wget \
        gnupg \
        ca-certificates 2>&1; then
        log_error "Errore durante l'installazione delle dipendenze base"
        log_error "Verifica che i repository siano configurati correttamente"
        log_error "Esegui manualmente: sudo apt-get install -y build-essential dkms linux-headers-$(uname -r)"
        exit 1
    fi
    
    log_info "Dipendenze installate"
}

# Installa driver NVIDIA
install_nvidia_driver() {
    log_info "Installazione driver NVIDIA..."
    
    # Verifica se i driver sono già installati
    if dpkg -l | grep -q "^ii.*nvidia-driver"; then
        local installed_driver=$(dpkg -l | grep "^ii.*nvidia-driver" | awk '{print $2}' | head -1)
        log_info "Driver NVIDIA già installato: $installed_driver"
        
        # Verifica se il driver funziona effettivamente
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
            log_info "Driver NVIDIA funzionante"
            return
        else
            log_warn "Driver installato ma non funzionante (probabilmente necessario riavvio o reinstallazione)"
            log_info "Tento reinstallazione del driver..."
            # Continua con l'installazione per assicurarsi che sia configurato correttamente
        fi
    fi
    
    # Pulisci repository NVIDIA/CUDA esistenti che potrebbero causare conflitti
    log_info "Pulizia repository NVIDIA/CUDA esistenti..."
    rm -f /etc/apt/sources.list.d/cuda*.list
    rm -f /etc/apt/sources.list.d/nvidia*.list
    rm -f /etc/apt/sources.list.d/*cuda*.list
    rm -f /etc/apt/sources.list.d/*nvidia*.list
    rm -f /usr/share/keyrings/cuda*.gpg
    rm -f /usr/share/keyrings/nvidia*.gpg
    
    # Aggiorna repository dopo pulizia
    apt-get update -qq 2>&1 | grep -v "WARNING\|Ign\|Hit" || true
    
    # Metodo 1: Prova con ubuntu-drivers autoinstall (funziona anche su Debian)
    if command -v ubuntu-drivers &> /dev/null; then
        log_info "Installazione driver NVIDIA tramite ubuntu-drivers autoinstall..."
        if ubuntu-drivers autoinstall; then
            log_info "Driver NVIDIA installati con successo tramite ubuntu-drivers"
            return
        else
            log_warn "ubuntu-drivers autoinstall fallito, provo metodo alternativo..."
        fi
    else
        log_info "ubuntu-drivers non disponibile, installo direttamente..."
    fi
    
    # Metodo 2: Installa driver NVIDIA direttamente dai repository Debian
    log_info "Installazione driver NVIDIA dai repository Debian..."
    if apt-get install -y nvidia-driver firmware-misc-nonfree; then
        log_info "Driver NVIDIA installati con successo"
    else
        log_error "Errore durante l'installazione dei driver"
        log_error "Prova a installare manualmente:"
        log_error "  sudo ubuntu-drivers autoinstall"
        log_error "  oppure: sudo apt-get install -y nvidia-driver"
        exit 1
    fi
}

# Installa CUDA Toolkit
install_cuda() {
    log_info "Installazione CUDA Toolkit..."
    
    # Verifica se CUDA è già installato
    if command -v nvcc &> /dev/null; then
        log_warn "CUDA Toolkit già installato"
        nvcc --version | head -3
        read -p "Vuoi reinstallare CUDA? (s/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Ss]$ ]]; then
            log_info "CUDA non verrà reinstallato"
            return
        fi
    fi
    
    log_info "Scarico e installo CUDA Toolkit da NVIDIA..."
    
    # Determina architettura
    local arch=$(dpkg --print-architecture)
    local os_version=$(lsb_release -sr)
    local os_codename=$(lsb_release -sc)
    
    log_info "Architettura: $arch, OS: $os_version ($os_codename)"
    
    # Installa CUDA dai repository NVIDIA (metodo consigliato)
    local cuda_repo_url="https://developer.download.nvidia.com/compute/cuda/repos"
    
    # Per Debian, usa il repository Ubuntu più compatibile o installa direttamente
    log_info "Installazione CUDA Toolkit tramite repository NVIDIA..."
    
    # Metodo alternativo: installa CUDA tramite pacchetti Debian se disponibili
    if apt-cache search cuda-toolkit | grep -q "^cuda-toolkit"; then
        log_info "Installazione CUDA Toolkit dai repository Debian..."
        if ! apt-get install -y cuda-toolkit; then
            log_warn "CUDA Toolkit non disponibile nei repository Debian"
            log_info "Installazione CUDA tramite repository NVIDIA..."
            install_cuda_from_nvidia
            local nvidia_result=$?
            # Se fallisce (codice 2 = fallback richiesto, o altro errore), usa fallback Debian
            if [ $nvidia_result -ne 0 ]; then
                install_cuda_from_debian
            fi
        fi
    else
        install_cuda_from_nvidia
        local nvidia_result=$?
        # Se fallisce (codice 2 = fallback richiesto, o altro errore), usa fallback Debian
        if [ $nvidia_result -ne 0 ]; then
            install_cuda_from_debian
        fi
    fi
}

# Installa CUDA direttamente da NVIDIA
install_cuda_from_nvidia() {
    log_info "Installazione CUDA Toolkit da repository NVIDIA..."
    
    local cuda_version="12.6"  # Versione CUDA stabile
    local os="ubuntu2404"  # Usa Ubuntu 24.04 repo per Debian 13 (compatibile)
    local arch=$(dpkg --print-architecture)
    
    if [ "$arch" = "amd64" ]; then
        arch="x86_64"
    fi
    
    log_info "Versione CUDA: $cuda_version"
    log_info "OS: $os, Arch: $arch"
    
    # Crea directory per keyring se non esiste
    mkdir -p /usr/share/keyrings
    
    # Aggiungi chiave GPG NVIDIA
    local gpg_key_url="https://developer.download.nvidia.com/compute/cuda/repos/${os}/${arch}/3bf863cc.pub"
    log_info "Aggiunta chiave GPG NVIDIA..."
    
    if wget -qO- "$gpg_key_url" 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg 2>/dev/null; then
        log_info "Chiave GPG aggiunta"
    else
        log_warn "Impossibile scaricare chiave GPG dal repository, provo metodo alternativo..."
        # Prova a scaricare direttamente
        if wget -q "$gpg_key_url" -O /tmp/cuda-key.pub 2>/dev/null; then
            gpg --dearmor < /tmp/cuda-key.pub > /usr/share/keyrings/cuda-keyring.gpg 2>/dev/null
            rm -f /tmp/cuda-key.pub
            log_info "Chiave GPG aggiunta (metodo alternativo)"
        else
            log_error "Errore nell'aggiunta della chiave GPG"
            log_info "Installazione CUDA manuale richiesta"
            log_info "Scarica da: https://developer.nvidia.com/cuda-downloads"
            return 1
        fi
    fi
    
    # Aggiungi repository CUDA
    local cuda_repo="deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/${os}/${arch} /"
    
    if [ -f /usr/share/keyrings/cuda-keyring.gpg ]; then
        echo "$cuda_repo" > /etc/apt/sources.list.d/cuda.list
        
        # Prova ad aggiornare repository e cattura errori
        local update_output=$(apt-get update -qq 2>&1)
        local update_status=$?
        
        if [ $update_status -eq 0 ] && ! echo "$update_output" | grep -qiE "non è firmato|not signed|signature verification failed|InRelease.*non è firmato"; then
            log_info "Repository CUDA aggiunto"
        else
            log_warn "Verifica firma repository NVIDIA fallita (problema SHA1 con chiave GPG)"
            log_warn "La chiave NVIDIA usa SHA1 che non è più accettato da Debian (da febbraio 2026)"
            log_info "Tento installazione senza verifica firma (solo per repository NVIDIA)..."
            
            # Rimuovi repository con firma e aggiungi senza verifica firma
            rm -f /etc/apt/sources.list.d/cuda.list
            echo "deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/${os}/${arch} /" > /etc/apt/sources.list.d/cuda.list
            
            if apt-get update -qq 2>&1 > /dev/null; then
                log_warn "Repository CUDA aggiunto senza verifica firma (solo per repository NVIDIA)"
                log_warn "Questo è necessario a causa del problema SHA1 con la chiave GPG di NVIDIA"
            else
                log_warn "Impossibile aggiungere repository CUDA NVIDIA"
                log_info "Uso repository Debian come fallback..."
                rm -f /etc/apt/sources.list.d/cuda.list
                # Non fare return, continua con installazione da Debian
                return 2  # Codice speciale per indicare fallback
            fi
        fi
    else
        log_warn "Chiave GPG non trovata, uso repository Debian come fallback"
        return 2  # Codice speciale per indicare fallback
    fi
    
    # Installa CUDA Toolkit
    log_info "Installazione CUDA Toolkit (questo può richiedere tempo e spazio ~3GB)..."
    
    # Prova prima con versione specifica, poi generica
    if apt-get install -y cuda-toolkit-${cuda_version//./-} 2>/dev/null; then
        log_info "CUDA Toolkit ${cuda_version} installato con successo"
    elif apt-get install -y cuda-toolkit 2>/dev/null; then
        log_info "CUDA Toolkit installato con successo"
    else
        log_warn "Installazione CUDA dai repository NVIDIA fallita"
        log_info "Tento installazione da repository Debian (versione più vecchia ma stabile)..."
        install_cuda_from_debian
    fi
}

# Installa CUDA dai repository Debian (fallback)
install_cuda_from_debian() {
    log_info "Installazione CUDA Toolkit dai repository Debian..."
    log_warn "Nota: questa è una versione più vecchia ma stabile"
    
    # Fallback: installa versione base da repository Debian
    if apt-get install -y nvidia-cuda-toolkit; then
        log_info "CUDA Toolkit base installato dai repository Debian"
        log_warn "Nota: questa è una versione più vecchia, per la versione più recente installa manualmente"
        log_info "Versione più recente disponibile su: https://developer.nvidia.com/cuda-downloads"
    else
        log_error "Installazione CUDA fallita completamente"
        log_info "Opzioni:"
        log_info "  1. Installa manualmente da: https://developer.nvidia.com/cuda-downloads"
        log_info "  2. Verifica che i repository non-free siano abilitati"
        return 1
    fi
}

# Configura variabili d'ambiente CUDA
configure_cuda_env() {
    log_info "Configurazione variabili d'ambiente CUDA..."
    
    local cuda_path="/usr/local/cuda"
    local bashrc_cuda=""
    
    # Verifica se CUDA è installato
    if [ -d "$cuda_path" ] || command -v nvcc &> /dev/null; then
        # Aggiungi a /etc/profile.d/cuda.sh
        cat > /etc/profile.d/cuda.sh <<'EOF'
# CUDA Environment Variables
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
        chmod +x /etc/profile.d/cuda.sh
        log_info "Variabili d'ambiente CUDA configurate in /etc/profile.d/cuda.sh"
        
        # Aggiungi anche a .bashrc se esiste
        if [ -f /root/.bashrc ]; then
            if ! grep -q "CUDA" /root/.bashrc; then
                echo "" >> /root/.bashrc
                echo "# CUDA Environment" >> /root/.bashrc
                echo "export PATH=/usr/local/cuda/bin:\$PATH" >> /root/.bashrc
                echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> /root/.bashrc
            fi
        fi
    else
        log_warn "CUDA non trovato in /usr/local/cuda, variabili d'ambiente non configurate"
    fi
}

# Verifica installazione
verify_installation() {
    echo ""
    log_info "=== Verifica installazione ==="
    echo ""
    
    # Verifica driver NVIDIA
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader &> /dev/null; then
            log_info "Driver NVIDIA:"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | \
                sed 's/^/  /'
            echo ""
        else
            log_warn "nvidia-smi trovato ma driver non funzionante"
            log_warn "Possibili cause:"
            log_warn "  1. Driver installato ma moduli kernel non caricati (riavvia il sistema)"
            log_warn "  2. Driver non compatibile con il kernel corrente"
            log_warn "  3. Problemi con Secure Boot (disabilita o configura)"
            echo ""
            # Verifica se i pacchetti sono installati
            if dpkg -l | grep -q "^ii.*nvidia-driver"; then
                local installed_driver=$(dpkg -l | grep "^ii.*nvidia-driver" | awk '{print $2}' | head -1)
                log_info "Driver installato: $installed_driver"
                log_info "Verifica moduli kernel (vedi sotto)..."
            fi
            echo ""
        fi
    else
        log_error "nvidia-smi non trovato - driver potrebbero non essere installati correttamente"
    fi
    
    # Verifica CUDA
    if command -v nvcc &> /dev/null; then
        log_info "CUDA Toolkit:"
        nvcc --version | head -4 | sed 's/^/  /'
        echo ""
    else
        log_warn "nvcc non trovato - CUDA potrebbe non essere installato"
        log_info "Verifica se CUDA è in /usr/local/cuda"
    fi
    
    # Verifica moduli kernel
    if lsmod | grep -q nvidia; then
        log_info "Moduli kernel NVIDIA caricati:"
        lsmod | grep nvidia | sed 's/^/  /'
    else
        log_warn "Moduli kernel NVIDIA non caricati"
        log_info "Potrebbe essere necessario riavviare il sistema"
    fi
}

# Helper per gestire PEP 668 (externally-managed environment)
# Rileva se Python è "externally managed" e restituisce i flag necessari per pip
get_pip_flags() {
    # Verifica se siamo su Debian 13 (trixie) che implementa PEP 668
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [ "$VERSION_CODENAME" = "trixie" ] || [ "$VERSION_ID" = "13" ]; then
            echo "--break-system-packages"
            return
        fi
    fi
    # Verifica versione Python (3.13+ ha PEP 668 di default)
    local python_version=$(python3 --version 2>&1 | sed -n 's/.*Python \([0-9]\+\.[0-9]\+\).*/\1/p')
    if [ -n "$python_version" ]; then
        # Confronta versione (3.13 o superiore) usando awk invece di bc
        local major=$(echo "$python_version" | cut -d. -f1)
        local minor=$(echo "$python_version" | cut -d. -f2)
        if [ "$major" -gt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -ge 13 ]); then
            echo "--break-system-packages"
            return
        fi
    fi
    echo ""
}

# Installa framework ML per inferenza
install_ml_frameworks() {
    log_info "Installazione framework ML per inferenza..."
    
    # Verifica Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 non trovato, installazione framework ML saltata"
        return 1
    fi
    
    # Verifica pip
    if ! command -v pip3 &> /dev/null; then
        log_info "pip3 non trovato, installazione pip..."
        apt-get install -y python3-pip > /dev/null 2>&1
    fi
    
    log_question "Quali framework ML vuoi installare?"
    echo "  1) PyTorch (consigliato per inferenza)"
    echo "  2) TensorFlow"
    echo "  3) ONNX Runtime (ottimizzato per inferenza)"
    echo "  4) vLLM (ottimizzato per LLM inference - richiede PyTorch)"
    echo "  5) PyTorch + vLLM (consigliato per LLM)"
    echo "  6) Tutti e tre (PyTorch, TensorFlow, ONNX)"
    echo "  7) Nessuno (salta)"
    read -p "Scelta [1-7] (default: 5): " -n 1 -r choice
    echo
    
    case ${choice:-5} in
        1)
            install_pytorch
            ;;
        2)
            install_tensorflow
            ;;
        3)
            install_onnxruntime
            ;;
        4)
            log_warn "vLLM richiede PyTorch, installo prima PyTorch..."
            install_pytorch
            install_vllm
            ;;
        5)
            install_pytorch
            install_vllm
            ;;
        6)
            install_pytorch
            install_tensorflow
            install_onnxruntime
            ;;
        7)
            log_info "Installazione framework ML saltata"
            ;;
        *)
            log_info "Scelta non valida, installo PyTorch + vLLM di default"
            install_pytorch
            install_vllm
            ;;
    esac
}

# Installa PyTorch con supporto CUDA
install_pytorch() {
    log_info "Installazione PyTorch con supporto CUDA..."
    
    # Verifica versione CUDA installata
    local cuda_version=""
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        log_info "CUDA versione rilevata: $cuda_version"
    fi
    
    # Installa PyTorch con CUDA (versione più recente disponibile)
    log_info "Installazione PyTorch (questo può richiedere tempo)..."
    
    # Verifica se Python è "externally managed" (PEP 668 - Debian 13/Python 3.13+)
    local pip_flags=$(get_pip_flags)
    if [ -n "$pip_flags" ]; then
        log_warn "Python è 'externally managed' (PEP 668), uso --break-system-packages"
        # Aggiungi --ignore-installed per evitare conflitti con pacchetti Debian
        pip_flags="$pip_flags --ignore-installed"
    fi
    
    # Prova prima con CUDA 12.x, poi 11.x come fallback
    if pip3 install $pip_flags torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>&1 | tee /tmp/pytorch_install.log; then
        log_info "PyTorch installato con supporto CUDA 12.1"
    elif pip3 install $pip_flags torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | tee /tmp/pytorch_install.log; then
        log_info "PyTorch installato con supporto CUDA 11.8"
    else
        log_warn "Installazione PyTorch con CUDA fallita, installo versione CPU"
        pip3 install $pip_flags torch torchvision torchaudio 2>&1 | tee /tmp/pytorch_install.log || {
            log_error "Errore durante l'installazione di PyTorch"
            return 1
        }
    fi
    
    # Verifica installazione
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA disponibile: {torch.cuda.is_available()}')" 2>/dev/null; then
        log_info "PyTorch installato correttamente"
    else
        log_warn "PyTorch installato ma verifica fallita"
    fi
}

# Installa TensorFlow con supporto CUDA
install_tensorflow() {
    log_info "Installazione TensorFlow con supporto CUDA..."
    
    local pip_flags=$(get_pip_flags)
    if [ -n "$pip_flags" ]; then
        log_warn "Python è 'externally managed' (PEP 668), uso --break-system-packages"
        # Aggiungi --ignore-installed per evitare conflitti con pacchetti Debian
        pip_flags="$pip_flags --ignore-installed"
    fi
    
    log_info "Installazione TensorFlow (questo può richiedere tempo)..."
    if pip3 install $pip_flags tensorflow[and-cuda] 2>&1 | tee /tmp/tensorflow_install.log; then
        log_info "TensorFlow installato con supporto CUDA"
    else
        log_warn "Installazione TensorFlow con CUDA fallita, installo versione base"
        pip3 install $pip_flags tensorflow 2>&1 | tee /tmp/tensorflow_install.log || {
            log_error "Errore durante l'installazione di TensorFlow"
            return 1
        }
    fi
    
    # Verifica installazione
    if python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}, GPU disponibile: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')" 2>/dev/null; then
        log_info "TensorFlow installato correttamente"
    else
        log_warn "TensorFlow installato ma verifica fallita"
    fi
}

# Installa ONNX Runtime con supporto CUDA
install_onnxruntime() {
    log_info "Installazione ONNX Runtime con supporto CUDA..."
    
    local pip_flags=$(get_pip_flags)
    if [ -n "$pip_flags" ]; then
        log_warn "Python è 'externally managed' (PEP 668), uso --break-system-packages"
        # Aggiungi --ignore-installed per evitare conflitti con pacchetti Debian
        pip_flags="$pip_flags --ignore-installed"
    fi
    
    log_info "Installazione ONNX Runtime (ottimizzato per inferenza)..."
    if pip3 install $pip_flags onnxruntime-gpu 2>&1 | tee /tmp/onnx_install.log; then
        log_info "ONNX Runtime GPU installato"
    else
        log_warn "Installazione ONNX Runtime GPU fallita, installo versione CPU"
        pip3 install $pip_flags onnxruntime 2>&1 | tee /tmp/onnx_install.log || {
            log_error "Errore durante l'installazione di ONNX Runtime"
            return 1
        }
    fi
    
    # Verifica installazione
    if python3 -c "import onnxruntime as ort; print(f'ONNX Runtime {ort.__version__}')" 2>/dev/null; then
        log_info "ONNX Runtime installato correttamente"
    else
        log_warn "ONNX Runtime installato ma verifica fallita"
    fi
}

# Installa vLLM per inferenza LLM ottimizzata
install_vllm() {
    log_info "Installazione vLLM (ottimizzato per LLM inference)..."
    
    # Verifica che PyTorch sia installato
    if ! python3 -c "import torch" 2>/dev/null; then
        log_error "PyTorch non trovato! vLLM richiede PyTorch."
        log_info "Installo PyTorch prima..."
        install_pytorch
    fi
    
    # Verifica che CUDA sia disponibile in PyTorch
    if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA non disponibile'" 2>/dev/null; then
        log_warn "CUDA non disponibile in PyTorch. vLLM richiede CUDA."
        log_warn "Assicurati di aver riavviato dopo l'installazione dei driver NVIDIA"
    fi
    
    log_info "Installazione vLLM (questo può richiedere molto tempo e spazio ~5-10GB)..."
    log_warn "vLLM richiede molto spazio su disco e tempo di installazione"
    
    local pip_flags=$(get_pip_flags)
    if [ -n "$pip_flags" ]; then
        log_warn "Python è 'externally managed' (PEP 668), uso --break-system-packages"
        # Aggiungi --ignore-installed per evitare conflitti con pacchetti Debian
        # Questo evita errori quando pip cerca di disinstallare pacchetti gestiti dal sistema
        pip_flags="$pip_flags --ignore-installed"
    fi
    
    # Installa vLLM con supporto CUDA
    # vLLM richiede una versione specifica di CUDA e PyTorch
    if pip3 install $pip_flags vllm 2>&1 | tee /tmp/vllm_install.log; then
        log_info "vLLM installato con successo"
    else
        log_error "Errore durante l'installazione di vLLM"
        log_info "Prova manualmente: pip3 install vllm"
        log_info "Oppure con build da sorgente se ci sono problemi di compatibilità"
        return 1
    fi
    
    # Verifica installazione
    if python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null; then
        log_info "vLLM installato correttamente"
        
        # Test rapido se CUDA è disponibile
        if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
            log_info "vLLM pronto per l'uso con GPU!"
        else
            log_warn "vLLM installato ma CUDA non disponibile - riavvia il sistema"
        fi
    else
        log_warn "vLLM installato ma verifica fallita"
    fi
}

# Mostra informazioni finali
show_final_info() {
    echo ""
    log_info "=== Installazione completata ==="
    echo ""
    
    # Verifica se nvidia-smi funziona prima di dare istruzioni
    local driver_working=false
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        driver_working=true
    fi
    
    if [ "$driver_working" = false ]; then
        log_warn "IMPORTANTE: Driver NVIDIA installato ma non funzionante!"
        log_warn "È NECESSARIO riavviare il sistema per caricare i moduli kernel NVIDIA"
        echo ""
    else
        log_warn "IMPORTANTE: Riavvia il sistema per completare l'installazione!"
        echo ""
    fi
    
    log_info "Dopo il riavvio, verifica con:"
    echo "  nvidia-smi"
    echo "  nvcc --version"
    echo ""
    
    # Verifica framework ML installati
    if python3 -c "import torch" 2>/dev/null; then
        log_info "PyTorch disponibile - testa con: python3 -c \"import torch; print(torch.cuda.is_available())\""
    fi
    if python3 -c "import tensorflow" 2>/dev/null; then
        log_info "TensorFlow disponibile - testa con: python3 -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
    fi
    if python3 -c "import onnxruntime" 2>/dev/null; then
        log_info "ONNX Runtime disponibile"
    fi
    if python3 -c "import vllm" 2>/dev/null; then
        log_info "vLLM disponibile - pronto per inferenza LLM!"
        echo ""
        log_info "Esempio uso vLLM:"
        echo "  from vllm import LLM"
        echo "  llm = LLM(model='mistralai/Mistral-7B-v0.1')"
        echo "  output = llm.generate('Hello, my name is')"
    fi
    
    echo ""
    log_info "Per fare inferenza:"
    echo "  1. Assicurati di aver riavviato il sistema"
    echo "  2. Verifica GPU: nvidia-smi"
    echo "  3. Carica il tuo modello e usa la GPU per l'inferenza"
    echo ""
    
    log_info "Esempio PyTorch:"
    echo "  import torch"
    echo "  device = 'cuda' if torch.cuda.is_available() else 'cpu'"
    echo "  model = model.to(device)"
    echo ""
    
    log_info "Se nvidia-smi non funziona dopo il riavvio:"
    echo "  1. Verifica: dmesg | grep -i nvidia"
    echo "  2. Verifica moduli: lsmod | grep nvidia"
    echo "  3. Riavvia servizi: sudo systemctl restart gdm3 (se usi GNOME)"
    echo ""
}

# Main
main() {
    log_info "=== Script di installazione driver NVIDIA con CUDA ==="
    echo ""
    
    check_root
    check_nvidia_gpu
    enable_nonfree_repos
    install_dependencies
    install_nvidia_driver
    install_cuda
    configure_cuda_env
    verify_installation
    install_ml_frameworks
    show_final_info
    
    log_info "Installazione completata!"
    log_warn "RICORDA: Riavvia il sistema per applicare le modifiche!"
}

# Esegui main
main "$@"
