#!/bin/bash
# Create .venv with torch 2.5.1 (as pinned in requirements.txt) for this machine: driver 550 / CUDA 12.4, Python 3.10.
# usage: bash scripts/setup_venv.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export PIP_CONFIG_FILE=/dev/null   # the global pip config points at pypi.ngc.nvidia.com, which does not resolve here
PY=/usr/bin/python3.10             # flash-attn has a prebuilt cp310 wheel for torch 2.5; conda's 3.14 has none
FLASH_ATTN_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

if [ ! -x .venv/bin/python ]; then
    # python3.10-venv (ensurepip) is not installed system-wide: create without pip, then bootstrap pip with the system pip
    $PY -m venv --without-pip .venv
    $PY -m pip --python .venv/bin/python install -q pip
fi
pip=(.venv/bin/python -m pip install -q)

"${pip[@]}" setuptools wheel   # triton (used by torch.compile) imports setuptools; a --without-pip venv lacks it
"${pip[@]}" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"${pip[@]}" "$FLASH_ATTN_WHL"   # prebuilt (torch2.5, cxx11abi=False like the PyPI torch wheel); `pip install flash_attn` would compile for ~1h
"${pip[@]}" -r requirements.txt huggingface_hub   # torch / flash_attn pins are already satisfied, so nothing is rebuilt
# opencv-python needs libGL, which this container lacks; the headless build provides the same cv2 module
.venv/bin/python -m pip uninstall -q -y opencv-python
"${pip[@]}" opencv-python-headless

.venv/bin/python - <<'EOF'
import torch, flash_attn, cv2, timm
assert torch.__version__.startswith('2.5.1'), torch.__version__
assert torch.cuda.is_available(), 'CUDA not available'
print(f'torch {torch.__version__} (cuda {torch.version.cuda}), flash_attn {flash_attn.__version__}, timm {timm.__version__}, cv2 {cv2.__version__}: OK')
EOF
