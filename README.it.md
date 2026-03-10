# vllm-cockpit

Guida semplice per configurare e avviare vLLM con Python 3.11.

> 🇮🇹 [Versione italiana](README.it.md) | 🇬🇧 [English version](README.md)

## Requisiti

- **Python 3.11** (richiesto)
- GPU NVIDIA (opzionale, ma consigliata)
- CUDA 12.1+ (per supporto GPU)
- Driver NVIDIA installati (necessari per GPU)

## Installazione Driver NVIDIA e CUDA

**IMPORTANTE**: Se non hai ancora installato i driver NVIDIA e CUDA, esegui prima lo script di installazione:

```bash
cd /home/gds/vllm
sudo ./install-nvidia-cuda.sh
```

Questo script:
- ✅ Verifica la presenza della scheda NVIDIA
- ✅ Installa i driver NVIDIA appropriati
- ✅ Installa CUDA Toolkit (versione 12.6 o compatibile)
- ✅ Configura le variabili d'ambiente CUDA
- ✅ Opzionalmente installa PyTorch e vLLM (se richiesto)

**Nota**: Dopo l'installazione dei driver NVIDIA, **è necessario riavviare il sistema** per caricare i moduli kernel.

Dopo il riavvio, verifica l'installazione:
```bash
nvidia-smi
nvcc --version
```

## Versioni Installate

- **vLLM**: 0.17.0 (compatibile con Qwen/Qwen3.5)
- **Transformers**: >=4.56.0,<5.0.0 (compatibile con vLLM 0.17.0)

## Setup Automatico

### 1. Installa Driver NVIDIA e CUDA (se necessario)

Se non hai ancora installato driver NVIDIA e CUDA:

```bash
cd /home/gds/vllm
sudo ./install-nvidia-cuda.sh
```

**Dopo l'installazione, riavvia il sistema** e verifica:
```bash
nvidia-smi
nvcc --version
```

### 2. Installa vLLM e dipendenze

Esegui lo script per configurare vLLM:

```bash
cd /home/gds/vllm
./setup.sh
```

Lo script:
- Verifica Python 3.11
- Crea ambiente virtuale
- Installa vLLM e dipendenze da `requirements.txt`
- Verifica l'installazione

## Setup Manuale

### 1. Crea ambiente virtuale con Python 3.11

```bash
cd /home/gds/vllm
python3.11 -m venv venv
```

### 2. Attiva l'ambiente virtuale e aggiorna pip

```bash
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 3. Installa vLLM e dipendenze

```bash
pip install -r requirements.txt
```

Il file `requirements.txt` contiene:
- `vllm==0.17.0` - Versione stabile compatibile con Qwen3.5
- `transformers>=4.56.0,<5.0.0` - Versione compatibile con vLLM 0.17.0
- `requests>=2.31.0` - Per le richieste HTTP

### 4. Verifica installazione

```bash
python -c "import vllm; import transformers; print('✅ vLLM:', vllm.__version__); print('✅ Transformers:', transformers.__version__)"
```

## Avvio Server

### Con Docker + Web UI (Consigliato)

Il Docker include una Web UI React per controllare vLLM con tutti i parametri configurabili. vLLM viene avviato in background dal backend e non dipende dalla Web UI.

```bash
# Avvio semplice
./start-docker.sh

# Oppure con docker-compose
docker-compose up -d --build

# Visualizza i log
docker-compose logs -f

# Ferma il server
docker-compose down
```

**Servizi disponibili:**
- 🌐 **Web UI**: http://localhost:3000 - Interfaccia per controllare vLLM
- 🔌 **Backend API**: http://localhost:3001 - API REST per controllare vLLM
- 🚀 **vLLM Server**: Avviato dinamicamente sulla porta configurata (default 8000)

**Vantaggi Docker:**
- ✅ Usa il venv esistente (no reinstallazione)
- ✅ Web UI React per configurare e avviare vLLM
- ✅ vLLM avviato in background (non dipende dalla UI)
- ✅ Tutti i parametri configurabili dalla UI
- ✅ Rete Docker isolata (`vllm-network`)
- ✅ Facile collegare altri servizi
- ✅ Supporto GPU automatico
- ✅ Gestione semplice con docker-compose

### Avvio diretto (senza Docker)

```bash
source venv/bin/activate
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host 0.0.0.0 --port 8000 --trust-remote-code
```

### Modelli disponibili

**Modelli Qwen (compatibili con vLLM 0.17.0):**
- `Qwen/Qwen3.5-0.6B-Instruct` - Modello molto piccolo (~1.5GB VRAM)
- `Qwen/Qwen3.5-1.5B-Instruct` - Modello piccolo (~3GB VRAM)
- `Qwen/Qwen3.5-3B-Instruct` - Modello medio (~6GB VRAM)
- `Qwen/Qwen3.5-7B-Instruct` - Modello grande (~14GB VRAM)

**Altri modelli supportati:**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Modello piccolo (~2GB VRAM)
- `microsoft/Phi-2` - Modello medio (~5GB VRAM)
- `meta-llama/Llama-2-7b-chat-hf` - Modello grande (~8GB VRAM)
- `Qwen/Qwen2-1.5B-Instruct` - Modello piccolo (~3GB VRAM)

## Web UI

La Web UI permette di:
- ⚙️ Configurare tutti i parametri di vLLM (modello, porta, GPU memory, ecc.)
- ▶️ Avviare vLLM con i parametri desiderati
- ⏹️ Fermare vLLM
- 📋 Visualizzare i log in tempo reale
- 📊 Monitorare lo stato del server

**Parametri configurabili:**
- Modello HuggingFace
- Host e Porta
- GPU Memory Utilization
- Max Model Length
- Tensor Parallel Size
- Max Num Seqs
- Max Batched Tokens
- Trust Remote Code

## Note

- Il server vLLM sarà disponibile sulla porta configurata (default 8000)
- L'API è compatibile con OpenAI
- Il primo avvio potrebbe richiedere tempo per scaricare il modello (i tensori vengono scaricati automaticamente)
- vLLM viene avviato in background dal backend, non dipende dalla Web UI
- Con Docker, la rete `vllm-network` permette di collegare altri servizi facilmente
- **vLLM 0.17.0** è compatibile con Qwen/Qwen3.5 e supporta la maggior parte dei modelli transformer standard da HuggingFace

## Collegare altri servizi alla rete Docker

Per collegare un altro container alla stessa rete:

```yaml
# Nel docker-compose.yml del tuo altro servizio
networks:
  - vllm-network

networks:
  vllm-network:
    external: true
```

Oppure:

```bash
docker network connect vllm-network nome-container
```

## Troubleshooting

### Problema GPU non supportata

Se hai una GPU vecchia (es. Quadro P2000 con compute capability sm_61), PyTorch moderno potrebbe non supportarla. In questo caso vLLM funzionerà ma non userà la GPU.

### Verifica GPU

```bash
nvidia-smi
```

### Verifica Python

```bash
python3.11 --version
```

### Verifica versioni installate

```bash
source venv/bin/activate
python -c "import vllm; import transformers; print(f'vLLM: {vllm.__version__}'); print(f'Transformers: {transformers.__version__}')"
```

### Problemi di compatibilità

Se riscontri problemi con transformers, assicurati di avere la versione corretta:

```bash
source venv/bin/activate
pip install "transformers>=4.56.0,<5.0.0" --upgrade
```

**Nota**: vLLM 0.17.0 richiede `transformers<5.0.0`, quindi non installare transformers 5.x.

## Licenza e Copyright

Questo progetto utilizza **vLLM** e altre librerie open-source. Si prega di notare quanto segue:

### Licenza vLLM

**vLLM** è rilasciato sotto licenza **Apache License 2.0**. Per maggiori informazioni, consulta:
- vLLM GitHub: https://github.com/vllm-project/vllm
- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

### Librerie con Licenze Copyleft

Questo progetto include dipendenze coperte da licenze copyleft (come GPL, LGPL, AGPL). Queste includono ma non si limitano a:

- **FlashAttention** - Licenziato sotto licenza BSD-style
- **PyTorch** - Licenziato sotto licenza BSD-style (con alcuni componenti sotto licenze diverse)
- **Transformers** - Licenziato sotto Apache License 2.0
- Varie librerie CUDA e dipendenze

**Importante**: Se distribuisci questo software o crei opere derivate, devi rispettare i termini di licenza di tutte le librerie incluse. Alcune licenze copyleft (come GPL) possono richiedere di rilasciare le tue modifiche sotto la stessa licenza.

Per un elenco completo delle dipendenze e delle loro licenze, controlla:
```bash
source venv/bin/activate
pip list --format=json
```

Oppure ispeziona i file di licenza nei pacchetti installati:
```bash
find venv/lib/python3.11/site-packages -name "LICENSE*" -o -name "COPYING*"
```

### Attribuzione

Questo progetto è costruito su:
- **vLLM** - Motore di inferenza LLM ad alta produttività
- **HuggingFace Transformers** - Libreria ML all'avanguardia
- **PyTorch** - Framework di deep learning
- E molti altri contributi open-source

Si prega di rispettare tutti gli avvisi di copyright e i termini di licenza quando si utilizza questo software.
