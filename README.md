# vllm-cockpit

Simple guide to configure and run vLLM with Python 3.11.

> 🇮🇹 [Versione italiana](README.it.md) | 🇬🇧 [English version](README.md)

## Requirements

- **Python 3.11** (required)
- NVIDIA GPU (optional, but recommended)
- CUDA 12.1+ (for GPU support)
- NVIDIA drivers installed (required for GPU)

## NVIDIA Driver and CUDA Installation

**IMPORTANT**: If you haven't installed NVIDIA drivers and CUDA yet, run the installation script first:

```bash
cd /home/gds/vllm
sudo ./install-nvidia-cuda.sh
```

This script:
- ✅ Verifies NVIDIA GPU presence
- ✅ Installs appropriate NVIDIA drivers
- ✅ Installs CUDA Toolkit (version 12.6 or compatible)
- ✅ Configures CUDA environment variables
- ✅ Optionally installs PyTorch and vLLM (if requested)

**Note**: After installing NVIDIA drivers, **you must reboot the system** to load the kernel modules.

After reboot, verify the installation:
```bash
nvidia-smi
nvcc --version
```

## Installed Versions

- **vLLM**: 0.17.0 (compatible with Qwen/Qwen3.5)
- **Transformers**: >=4.56.0,<5.0.0 (compatible with vLLM 0.17.0)

## Automatic Setup

### 1. Install NVIDIA Drivers and CUDA (if needed)

If you haven't installed NVIDIA drivers and CUDA yet:

```bash
cd /home/gds/vllm
sudo ./install-nvidia-cuda.sh
```

**After installation, reboot the system** and verify:
```bash
nvidia-smi
nvcc --version
```

### 2. Install vLLM and dependencies

Run the script to configure vLLM:

```bash
cd /home/gds/vllm
./setup.sh
```

The script:
- Verifies Python 3.11
- Creates virtual environment
- Installs vLLM and dependencies from `requirements.txt`
- Verifies the installation

## Manual Setup

### 1. Create virtual environment with Python 3.11

```bash
cd /home/gds/vllm
python3.11 -m venv venv
```

### 2. Activate virtual environment and update pip

```bash
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 3. Install vLLM and dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:
- `vllm==0.17.0` - Stable version compatible with Qwen3.5
- `transformers>=4.56.0,<5.0.0` - Version compatible with vLLM 0.17.0
- `requests>=2.31.0` - For HTTP requests

### 4. Verify installation

```bash
python -c "import vllm; import transformers; print('✅ vLLM:', vllm.__version__); print('✅ Transformers:', transformers.__version__)"
```

## Server Startup

### With Docker + Web UI (Recommended)

Docker includes a React Web UI to control vLLM with all configurable parameters. vLLM is started in the background by the backend and doesn't depend on the Web UI.

```bash
# Simple startup
./start-docker.sh

# Or with docker-compose
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop server
docker-compose down
```

**Available services:**
- 🌐 **Web UI**: http://localhost:3000 - Interface to control vLLM
- 🔌 **Backend API**: http://localhost:3001 - REST API to control vLLM
- 🚀 **vLLM Server**: Started dynamically on configured port (default 8000)

**Docker advantages:**
- ✅ Uses existing venv (no reinstallation)
- ✅ React Web UI to configure and start vLLM
- ✅ vLLM started in background (doesn't depend on UI)
- ✅ All parameters configurable from UI
- ✅ Isolated Docker network (`vllm-network`)
- ✅ Easy to connect other services
- ✅ Automatic GPU support
- ✅ Simple management with docker-compose

### Direct startup (without Docker)

```bash
source venv/bin/activate
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host 0.0.0.0 --port 8000 --trust-remote-code
```

### Available Models

**Qwen models (compatible with vLLM 0.17.0):**
- `Qwen/Qwen3.5-0.6B-Instruct` - Very small model (~1.5GB VRAM)
- `Qwen/Qwen3.5-1.5B-Instruct` - Small model (~3GB VRAM)
- `Qwen/Qwen3.5-3B-Instruct` - Medium model (~6GB VRAM)
- `Qwen/Qwen3.5-7B-Instruct` - Large model (~14GB VRAM)

**Other supported models:**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - Small model (~2GB VRAM)
- `microsoft/Phi-2` - Medium model (~5GB VRAM)
- `meta-llama/Llama-2-7b-chat-hf` - Large model (~8GB VRAM)
- `Qwen/Qwen2-1.5B-Instruct` - Small model (~3GB VRAM)

## Web UI

The Web UI allows you to:
- ⚙️ Configure all vLLM parameters (model, port, GPU memory, etc.)
- ▶️ Start vLLM with desired parameters
- ⏹️ Stop vLLM
- 📋 View logs in real-time
- 📊 Monitor server status

**Configurable parameters:**
- HuggingFace Model
- Host and Port
- GPU Memory Utilization
- Max Model Length
- Tensor Parallel Size
- Max Num Seqs
- Max Batched Tokens
- Trust Remote Code

## Notes

- The vLLM server will be available on the configured port (default 8000)
- The API is OpenAI compatible
- First startup may take time to download the model (tensors are downloaded automatically)
- vLLM is started in background by the backend, doesn't depend on the Web UI
- With Docker, the `vllm-network` allows easy connection of other services
- **vLLM 0.17.0** is compatible with Qwen/Qwen3.5 and supports most standard transformer models from HuggingFace

## Connect other services to Docker network

To connect another container to the same network:

```yaml
# In your other service's docker-compose.yml
networks:
  - vllm-network

networks:
  vllm-network:
    external: true
```

Or:

```bash
docker network connect vllm-network container-name
```

## Troubleshooting

### Unsupported GPU issue

If you have an old GPU (e.g. Quadro P2000 with compute capability sm_61), modern PyTorch might not support it. In this case vLLM will work but won't use the GPU.

### Verify GPU

```bash
nvidia-smi
```

### Verify Python

```bash
python3.11 --version
```

### Verify installed versions

```bash
source venv/bin/activate
python -c "import vllm; import transformers; print(f'vLLM: {vllm.__version__}'); print(f'Transformers: {transformers.__version__}')"
```

### Compatibility issues

If you encounter problems with transformers, make sure you have the correct version:

```bash
source venv/bin/activate
pip install "transformers>=4.56.0,<5.0.0" --upgrade
```

**Note**: vLLM 0.17.0 requires `transformers<5.0.0`, so don't install transformers 5.x.

## License and Copyright

This project uses **vLLM** and other open-source libraries. Please note the following:

### vLLM License

**vLLM** is licensed under the **Apache License 2.0**. For more information, see:
- vLLM GitHub: https://github.com/vllm-project/vllm
- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

### Libraries with Copyleft Licenses

This project includes dependencies that are covered by copyleft licenses (such as GPL, LGPL, AGPL). These include but are not limited to:

- **FlashAttention** - Licensed under BSD-style license
- **PyTorch** - Licensed under BSD-style license (with some components under different licenses)
- **Transformers** - Licensed under Apache License 2.0
- Various CUDA libraries and dependencies

**Important**: If you distribute this software or create derivative works, you must comply with the license terms of all included libraries. Some copyleft licenses (like GPL) may require you to release your modifications under the same license.

For a complete list of dependencies and their licenses, check:
```bash
source venv/bin/activate
pip list --format=json
```

Or inspect the license files in the installed packages:
```bash
find venv/lib/python3.11/site-packages -name "LICENSE*" -o -name "COPYING*"
```

### Attribution

This project is built on top of:
- **vLLM** - High-throughput LLM inference engine
- **HuggingFace Transformers** - State-of-the-art ML library
- **PyTorch** - Deep learning framework
- And many other open-source contributions

Please respect all copyright notices and license terms when using this software.
