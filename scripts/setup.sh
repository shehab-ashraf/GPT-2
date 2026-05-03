#!/bin/bash
# Setup script for Lightning AI GPU instances (training + eval)
set -e

pip install --upgrade pip --quiet

pip install \
    tiktoken datasets wandb huggingface_hub \
    safetensors ninja packaging tqdm \
    transformers accelerate \
    --quiet

# Install Flash Attention from pre-built wheel (falls back to source build)
PY=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
PT=$(python3 -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'torch{v[0]}.{v[1]}')")
CU=$(python3 -c "import torch; print(f'cu{torch.version.cuda.split(\".\")[0]}')")
ABI=$(python3 -c "import torch; print('cxx11abiTRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'cxx11abiFALSE')")
VER="2.7.3"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${VER}/flash_attn-${VER}+${CU}${PT}${ABI}-${PY}-${PY}-linux_x86_64.whl"

pip install "$URL" --quiet 2>/dev/null || pip install flash-attn --no-build-isolation

# HuggingFace auth (for model uploads)
[ -n "${HF_TOKEN:-}" ] && huggingface-cli login --token "$HF_TOKEN"

echo "Setup done."