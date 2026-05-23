#!/bin/bash
#SBATCH --job-name=CIC18__split__v1__RobustScaler_RUS_SMOTE_pca_logreg
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --mem=200GB
#SBATCH --output=/home/inginf/u32902122/TFG/04_experimentos/logs/out/output_CIC18__split__v1__RobustScaler_RUS_SMOTE_pca_logreg.log
#SBATCH --mail-user=javier.labradormunoz@alum.uca.es
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

set -e

NOMBRE_SCRIPT="CIC18__split__v1__RobustScaler_RUS_SMOTE_pca_logreg"
PATH_SCRIPT="03_codigo/CIC18/03_RobustScaler_logreg/CIC18__split__v1__RobustScaler_RUS_SMOTE_pca_logreg"

# ========= CONFIGURACIÓN DE RUTAS =========
# Si quieres, cambia BASE_DIR por la ruta de tu nuevo equipo.
# Si no existe, usa automáticamente el directorio actual.
BASE_DIR_DEFAULT="/home/inginf/u32902122/TFG"
if [ -d "$BASE_DIR_DEFAULT" ]; then
    BASE_DIR="$BASE_DIR_DEFAULT"
else
    BASE_DIR="$(pwd)"
fi

# ========= MODO SLURM / LOCAL =========
# Si hay SLURM_JOB_ID, estamos en sbatch.
# Si no, generamos uno para modo local.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    JOB_ID="$(date +"%Y%m%d_%H%M%S")"
else
    JOB_ID="$SLURM_JOB_ID"
fi

# Si hay SLURM_SUBMIT_DIR, usarlo. Si no, usar BASE_DIR o pwd.
if [ -z "${SLURM_SUBMIT_DIR:-}" ]; then
    SUBMIT_DIR="$BASE_DIR"
else
    SUBMIT_DIR="$SLURM_SUBMIT_DIR"
fi

BASE="$BASE_DIR/$PATH_SCRIPT"
NOTEBOOK="$BASE.ipynb"

NOTEBOOK_OUT_DIR="$BASE_DIR/04_experimentos/logs/notebooks"
LOG_OUT_DIR="$BASE_DIR/04_experimentos/logs/out"
NVIDIA_OUT_DIR="$BASE_DIR/04_experimentos/logs/nvidia"

NOTEBOOK_OUT_NAME="${NOMBRE_SCRIPT}_executed_${JOB_ID}.ipynb"
LOG_OUT_FILE="$LOG_OUT_DIR/output_${NOMBRE_SCRIPT}_${JOB_ID}.log"

CSV_NVIDIA="$NVIDIA_OUT_DIR/${NOMBRE_SCRIPT}_${JOB_ID}_nvidia-smi.csv"
TXT_NVIDIA="$NVIDIA_OUT_DIR/${NOMBRE_SCRIPT}_${JOB_ID}_nvidia-smi.txt"

mkdir -p "$NOTEBOOK_OUT_DIR" "$LOG_OUT_DIR" "$NVIDIA_OUT_DIR"

# ========= CONDA =========
# En SLURM puede existir module. En local normalmente no pasa nada si falla.
if command -v module >/dev/null 2>&1; then
    module load Anaconda3/2024.02-1 || true
fi

# Inicializar conda de forma robusta
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "No encuentro conda.sh"
    exit 1
fi

conda activate tfgClean
export PYTHONNOUSERSITE=1

# ========= EJECUCIÓN =========
cd "$SUBMIT_DIR"

NVIDIA_PID=""

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$TXT_NVIDIA" || true

    nvidia-smi -lms 1000 \
      --query-gpu=timestamp,pstate,power.management,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,temperature.gpu,temperature.memory,memory.used,memory.total,memory.free,clocks.current.sm,clocks.current.memory \
      --format=csv,nounits \
      -f "$CSV_NVIDIA" &
    NVIDIA_PID=$!
fi

cleanup() {
    if [ -n "$NVIDIA_PID" ]; then
        kill "$NVIDIA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

jupyter nbconvert \
  --execute \
  --to notebook \
  --ExecutePreprocessor.kernel_name=tfgClean \
  --ExecutePreprocessor.timeout=86400 \
  --output "$NOTEBOOK_OUT_NAME" \
  --output-dir "$NOTEBOOK_OUT_DIR" \
  "$NOTEBOOK" \
  > "$LOG_OUT_FILE" 2>&1
