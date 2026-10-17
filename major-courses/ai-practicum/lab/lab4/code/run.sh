#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/tools/env.sh"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Run bash setup.sh first, or set PYTHON_BIN to a prepared Python 3.10 environment.\n' >&2
  exit 1
fi
MODE="${1:-eval}"
case "$MODE" in
  init|check|prepare|plot)
    exec "$PYTHON_BIN" tools/portable.py "$MODE"
    ;;
  eval|train|smoke) ;;
  *) printf 'Usage: bash run.sh [init|check|prepare|smoke|eval|train|plot]\n' >&2; exit 2 ;;
esac
command -v tmux >/dev/null || { printf 'tmux is required (installed by setup.sh).\n' >&2; exit 1; }
nvidia-smi
export GPU_INDEX="${GPU_INDEX:-0}"
GPU_USED=$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits)
GPU_PIDS=$(nvidia-smi --id="$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$GPU_PIDS" ]] || (( GPU_USED > 1024 )); then
  printf 'GPU %s already uses %s MiB; select an idle GPU with GPU_INDEX. No process was stopped.\n' "$GPU_INDEX" "$GPU_USED" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
STAMP=$(date +%Y%m%d-%H%M%S)
export RUN_ID="${RUN_ID:-$STAMP-$MODE}"
LAB_NUMBER=$("$PYTHON_BIN" -c 'import json;print(json.load(open("package_config.json"))["lab"])')
if [[ "$LAB_NUMBER" == 3 ]]; then EXPERIMENT=arithmetic_model_optimization; else EXPERIMENT=arithmetic_model_acceleration; fi
LOG_DIR="$LAB_ROOT/experiments/$EXPERIMENT/runs/$RUN_ID-orchestrator"
mkdir -p "$LOG_DIR"
export LOG_DIR
export LAB_MODE="$MODE"
SESSION="lab${LAB_NUMBER}_${RUN_ID//[^a-zA-Z0-9_]/_}"
export SESSION
"$PYTHON_BIN" tools/portable.py launch
printf 'Started tmux session %s\nLog: %s/launcher.log\n' "$SESSION" "$LOG_DIR"
