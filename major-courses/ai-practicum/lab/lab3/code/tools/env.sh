#!/usr/bin/env bash
export LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-$LAB_ROOT/.conda/envs/wjz/bin/python}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$LAB_ROOT/cache"
export TMPDIR="$LAB_ROOT/cache/tmp"
export MPLCONFIGDIR="$LAB_ROOT/cache/matplotlib"
export YOLO_CONFIG_DIR="$LAB_ROOT/cache/ultralytics"
export TORCH_HOME="$LAB_ROOT/cache/torch"
export HF_HOME="$LAB_ROOT/cache/huggingface"
export PADDLE_HOME="$LAB_ROOT/cache/paddle"
export SWANLAB_ROOT="$LAB_ROOT/cache/swanlab"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
cd "$LAB_ROOT"
