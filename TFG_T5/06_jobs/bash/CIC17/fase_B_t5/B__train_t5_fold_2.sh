#!/bin/bash
#SBATCH --job-name=CIC17_B_T5_F2
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --mem=200GB
#SBATCH --output=06_jobs/logs/CIC17/fase_B_t5/slurm_%x_%j.out
#SBATCH --mail-user=javier.labradormunoz@alum.uca.es
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

set -euo pipefail

FOLD=2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    BASE_DIR="${TFG_BASE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
    RUNNER="$BASE_DIR/06_jobs/bash/CIC17/fase_B_t5/_run_t5_fold.sh"
else
    RUNNER="$SCRIPT_DIR/_run_t5_fold.sh"
fi

source "$RUNNER"
