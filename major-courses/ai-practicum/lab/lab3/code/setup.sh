#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/tools/env.sh"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_PKGS_DIRS="$LAB_ROOT/cache/conda" conda create -y -p "$LAB_ROOT/.conda/envs/wjz" --override-channels -c conda-forge python=3.10 pip tmux
  else
    mkdir -p cache/micromamba
    curl -fL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest -o cache/micromamba.tar.bz2
    tar -xjf cache/micromamba.tar.bz2 -C cache/micromamba bin/micromamba
    MAMBA_ROOT_PREFIX="$LAB_ROOT/cache/mamba" cache/micromamba/bin/micromamba create -y -p "$LAB_ROOT/.conda/envs/wjz" -c conda-forge python=3.10 pip tmux
  fi
fi
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
"$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools
"$PYTHON_BIN" -m pip install numpy==1.26.4
"$PYTHON_BIN" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PYTHON_BIN" -m pip install -r requirements.txt
if [[ "$("$PYTHON_BIN" -c 'import json;print(json.load(open("package_config.json"))["lab"])')" == 3 ]]; then
  "$PYTHON_BIN" -m pip install paddlepaddle-gpu==3.0.0 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
else
  "$PYTHON_BIN" -m pip install tensorrt-cu12==10.0.1 onnx==1.23.2
fi
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps opencv-python-headless==4.11.0.86
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" tools/portable.py init
if [[ "$("$PYTHON_BIN" -c 'import json;print(json.load(open("package_config.json"))["kind"])')" == cloud ]]; then
  "$PYTHON_BIN" tools/portable.py check
fi
printf 'Setup complete. Run: bash run.sh smoke; bash run.sh eval\n'
