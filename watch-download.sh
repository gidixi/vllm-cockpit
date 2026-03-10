#!/bin/bash
# Script inline per monitorare il download dei tensori nel container Docker
# Uso: ./watch-download.sh

echo "📥 Monitoraggio download tensori nel container Docker..."
echo "Premi Ctrl+C per uscire"
echo ""

# Verifica che il container esista
if ! docker ps --format '{{.Names}}' | grep -q "vllm-backend"; then
    echo "❌ Container vllm-backend non trovato!"
    echo "Avvia prima i container con: ./start-docker.sh"
    exit 1
fi

# Mostra i log filtrati per download/tensori
docker logs -f vllm-backend 2>&1 | grep --line-buffered -iE "download|downloading|huggingface|hf|tensor|model|progress|%|MB|GB|cache|safetensors" | while IFS= read -r line; do
    # Colora le righe in base al contenuto
    if echo "$line" | grep -qiE "error|fail|exception"; then
        echo -e "\033[0;31m$line\033[0m"  # Rosso per errori
    elif echo "$line" | grep -qiE "complete|done|finished|100%"; then
        echo -e "\033[0;32m$line\033[0m"  # Verde per completamento
    elif echo "$line" | grep -qiE "progress|%|downloading"; then
        echo -e "\033[0;33m$line\033[0m"  # Giallo per progresso
    else
        echo "$line"  # Normale per altri messaggi
    fi
done
