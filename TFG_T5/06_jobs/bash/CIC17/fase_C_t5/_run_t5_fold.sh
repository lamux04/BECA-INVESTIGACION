#!/bin/bash

if [ -z "${FOLD:-}" ]; then
    echo "FOLD no esta definido."
    exit 1
fi

if [ -n "${SLURM_JOB_ID:-}" ]; then
    BASE_DIR="${TFG_BASE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
else
    BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi

if [ ! -d "$BASE_DIR/04_experiments" ] || [ ! -d "$BASE_DIR/06_jobs" ]; then
    echo "La raiz del proyecto no es valida: $BASE_DIR"
    echo "Ejecuta sbatch desde TFG_T5."
    exit 1
fi

PYTHON_SCRIPT="$BASE_DIR/04_experiments/CIC17/fase_C_t5/C__train_t5_fold.py"
MODEL_PATH="$BASE_DIR/07_models/pretrained/t5-small"
LOG_DIR="$BASE_DIR/06_jobs/logs/CIC17/fase_C_t5"
JOB_ID="${SLURM_JOB_ID:-$(date +"%Y%m%d_%H%M%S")}"
LOG_FILE="$LOG_DIR/output_C__train_t5_fold_${FOLD}_${JOB_ID}.log"
NVIDIA_CSV="$LOG_DIR/nvidia_C__train_t5_fold_${FOLD}_${JOB_ID}.csv"
NVIDIA_TXT="$LOG_DIR/nvidia_C__train_t5_fold_${FOLD}_${JOB_ID}.txt"

mkdir -p "$LOG_DIR"

if command -v module >/dev/null 2>&1; then
    module load Anaconda3/2024.02-1 || true
fi

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "No se pudo inicializar Conda tras cargar el modulo Anaconda3."
    exit 1
fi

conda activate tfgClean
export PYTHONNOUSERSITE=1
cd "$BASE_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No se ha encontrado nvidia-smi. El entrenamiento requiere GPU."
    exit 1
fi

nvidia-smi > "$NVIDIA_TXT"

echo "Comprobando acceso de PyTorch a la GPU..."

if ! python - <<'PY'
import os
import sys

import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA de PyTorch: {torch.version.cuda}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit(1)
PY
then
    echo
    echo "PyTorch no puede usar la GPU asignada por SLURM."
    echo "El entorno tfgClean tiene una build de PyTorch incompatible con el driver del nodo."
    echo "En el nodo de acceso, activa tfgClean e instala la build oficial CUDA 12.6:"
    echo
    echo "  conda activate tfgClean"
    echo "  python -m pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126"
    echo
    echo "Despues comprueba que torch.version.cuda sea 12.6 y torch.cuda.is_available() sea True."
    exit 1
fi

nvidia-smi -lms 1000 \
    --query-gpu=timestamp,pstate,power.draw,temperature.gpu,memory.used,memory.total,memory.free,clocks.current.sm,clocks.current.memory \
    --format=csv,nounits \
    -f "$NVIDIA_CSV" &
NVIDIA_PID=$!

cleanup() {
    kill "$NVIDIA_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Entrenando T5 fase C con el fold $FOLD..."
echo "Modelo base: $MODEL_PATH"
echo "Log: $LOG_FILE"

python "$PYTHON_SCRIPT" \
    --fold "$FOLD" \
    --model-path "$MODEL_PATH" \
    2>&1 | tee "$LOG_FILE"
