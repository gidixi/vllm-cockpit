#!/bin/bash

# Script per configurare vLLM con Python 3.11
# Esegui: ./setup.sh

set -euo pipefail

VENV_DIR="$(dirname "$0")/venv"

echo "🔧 Setup vLLM con Python 3.11"
echo ""

# Verifica Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Errore: Python 3.11 non trovato!"
    echo "Installa Python 3.11 prima di continuare."
    exit 1
fi

echo "✅ Python 3.11 trovato: $(python3.11 --version)"
echo ""

# Crea venv se non esiste
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Ambiente virtuale già esistente in $VENV_DIR"
    read -p "Vuoi ricrearlo? (s/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[SsYy]$ ]]; then
        echo "🗑️  Rimozione venv esistente..."
        rm -rf "$VENV_DIR"
    else
        echo "📦 Uso venv esistente"
        echo ""
        echo "Per attivarlo:"
        echo "  source venv/bin/activate"
        exit 0
    fi
fi

# Crea ambiente virtuale
echo "📦 Creazione ambiente virtuale con Python 3.11..."
python3.11 -m venv "$VENV_DIR"

# Attiva venv
echo "🔌 Attivazione ambiente virtuale..."
source "$VENV_DIR/bin/activate"

# Aggiorna pip
echo "⬆️  Aggiornamento pip..."
pip install --upgrade pip setuptools wheel --quiet

# Installa vLLM e dipendenze da requirements.txt
REQUIREMENTS_FILE="$(dirname "$0")/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Errore: requirements.txt non trovato!"
    exit 1
fi

echo "📥 Installazione vLLM e dipendenze da requirements.txt (questo richiederà alcuni minuti)..."
pip install -r "$REQUIREMENTS_FILE"

# Verifica installazione
echo ""
echo "🔍 Verifica installazione..."
if python -c "import vllm; import transformers; print('✅ vLLM installato! Versione:', vllm.__version__); print('✅ Transformers installato! Versione:', transformers.__version__)" 2>/dev/null; then
    echo ""
    echo "✅ Setup completato con successo!"
    echo ""
    echo "Per avviare il server:"
    echo "  source venv/bin/activate"
    echo "  vllm serve Qwen/Qwen3.5-1.5B-Instruct --host 0.0.0.0 --port 8000 --trust-remote-code"
    echo ""
    echo "Oppure usa Docker con Web UI:"
    echo "  ./start-docker.sh"
else
    echo "❌ Errore: vLLM non installato correttamente"
    exit 1
fi
