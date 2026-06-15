#!/bin/bash
#SBATCH --job-name=CIC17_SPLIT
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --mem=200GB
#SBATCH --output=06_jobs/logs/CIC17/preprocessing/slurm_%x_%j.out
#SBATCH --mail-user=javier.labradormunoz@alum.uca.es
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

set -euo pipefail

# ========= RUTAS =========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    BASE_DIR="${TFG_BASE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
else
    BASE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
fi

if [ ! -d "$BASE_DIR/03_preprocessing" ] || [ ! -d "$BASE_DIR/06_jobs" ]; then
    echo "La raiz del proyecto no es valida: $BASE_DIR"
    echo "Ejecuta sbatch desde TFG_T5."
    exit 1
fi

PYTHON_SCRIPT="$BASE_DIR/03_preprocessing/CIC17/02_split_dataset.py"
NOMBRE_SCRIPT="$(basename "$PYTHON_SCRIPT" .py)"
LOG_OUT_DIR="$BASE_DIR/06_jobs/logs/CIC17/preprocessing"

JOB_ID="${SLURM_JOB_ID:-$(date +"%Y%m%d_%H%M%S")}"
LOG_OUT_FILE="$LOG_OUT_DIR/output_${NOMBRE_SCRIPT}_${JOB_ID}.log"

mkdir -p "$LOG_OUT_DIR"

# ========= CONDA =========
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

# ========= EJECUCION =========
cd "$BASE_DIR"

echo "Ejecutando division train/test y generacion de folds del dataset CIC17..."
echo "Script: $PYTHON_SCRIPT"
echo "Log: $LOG_OUT_FILE"

python "$PYTHON_SCRIPT" 2>&1 | tee "$LOG_OUT_FILE"

echo "Proceso finalizado correctamente."
