#!/usr/bin/env python3
"""
Script per scaricare modelli HuggingFace nella cache condivisa con vLLM.
I modelli scaricati saranno immediatamente disponibili per vLLM nel container.
"""

import os
import sys
from huggingface_hub import snapshot_download

# Cache condivisa con il container vLLM
CACHE_DIR = os.path.expanduser('~/.cache/huggingface')

def download_model(model_id, token=None, force_download=False):
    """
    Scarica un modello HuggingFace nella cache condivisa.
    
    Args:
        model_id: ID del modello (es. "Qwen/Qwen2.5-7B-Instruct")
        token: Token HuggingFace (opzionale, per modelli privati)
        force_download: Se True, forza il re-download anche se già presente
    """
    try:
        print(f"Scaricando {model_id} in {CACHE_DIR}...")
        print(f"Cache directory: {CACHE_DIR}")
        
        snapshot_download(
            repo_id=model_id,
            token=token or os.environ.get('HF_TOKEN'),
            cache_dir=CACHE_DIR,
            resume_download=not force_download,
            force_download=force_download
        )
        
        print(f"✅ Modello {model_id} scaricato con successo!")
        print(f"📁 Disponibile in: {CACHE_DIR}/hub/models--{model_id.replace('/', '--')}")
        print(f"🐳 Accessibile nel container vLLM come: /mnt/huggingface-gds/hub/models--{model_id.replace('/', '--')}")
        
    except Exception as e:
        print(f"❌ Errore nel download: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python3 download_model.py <model_id> [--token TOKEN] [--force]")
        print("\nEsempi:")
        print("  python3 download_model.py Qwen/Qwen2.5-7B-Instruct")
        print("  python3 download_model.py meta-llama/Llama-3.1-8B --token hf_xxxxx")
        print("  python3 download_model.py Qwen/Qwen2.5-7B-Instruct --force")
        sys.exit(1)
    
    model_id = sys.argv[1]
    token = None
    force = False
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--token' and i + 1 < len(sys.argv):
            token = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--force':
            force = True
            i += 1
        else:
            i += 1
    
    download_model(model_id, token, force)
