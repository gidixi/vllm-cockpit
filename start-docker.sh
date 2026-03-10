#!/bin/bash

# Script per avviare vLLM con Docker e Web UI
# Esegui: ./start-docker.sh

set -euo pipefail

echo "🐳 Avvio vLLM con Docker e Web UI"
echo ""

# Verifica che il venv esista
if [ ! -d "venv" ]; then
    echo "❌ Errore: venv non trovato!"
    echo "Esegui prima: ./setup.sh"
    exit 1
fi

# Verifica che vLLM sia installato nel venv
if [ ! -f "venv/bin/vllm" ]; then
    echo "❌ Errore: vLLM non installato nel venv!"
    echo "Esegui prima: ./setup.sh"
    exit 1
fi

# Avvia con docker-compose
echo "🚀 Avvio container (backend + frontend)..."
docker compose up -d --build

echo ""
echo "✅ Servizi avviati!"
echo ""
echo "🌐 Web UI disponibile su: http://localhost:3000"
echo "🔌 Backend API disponibile su: http://localhost:3001"
echo ""
echo "Usa la Web UI per avviare/fermare vLLM con i parametri desiderati"
echo ""
echo "Per vedere i log:"
echo "  docker-compose logs -f"
echo ""
echo "Per fermare:"
echo "  docker-compose down"
