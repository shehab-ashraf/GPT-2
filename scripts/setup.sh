#!/bin/bash
set -e

echo "Setting up Lightning AI environment..."

pip install --upgrade pip --quiet

echo "Installing core packages..."
pip install \
    tiktoken \
    datasets \
    wandb \
    huggingface_hub \
    safetensors \
    ninja \
    packaging \
    tqdm \
    --quiet

echo "Installing Flash Attention..."
PY=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
PT=$(python3 -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'torch{v[0]}.{v[1]}')")
CU=$(python3 -c "import torch; print(f'cu{torch.version.cuda.split(\".\")[0]}')")
VER="2.7.3"
ABI=$(python3 -c "import torch; print('cxx11abiTRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'cxx11abiFALSE')")
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${VER}/flash_attn-${VER}+${CU}${PT}${ABI}-${PY}-${PY}-linux_x86_64.whl"

pip install "$URL" --quiet 2>/dev/null || pip install flash-attn --no-build-isolation

echo "All done! Ready to train."