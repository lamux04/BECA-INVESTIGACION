#!/bin/bash
#SBATCH --job-name=run_pd_CIC17__RUS_SMOTE__v1__training_v1__3_5
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --mem=200GB
#SBATCH --output=/home/inginf/u32902122/TFG/04_experimentos/logs/out/output_run_pd_CIC17__RUS_SMOTE__v1__training_v1__3_5.log
#SBATCH --mail-user=javier.labradormunoz@alum.uca.es
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

NOMBRE_SCRIPT="run_pd_CIC17__RUS_SMOTE__v1__training_v1__3_5"
PATH_SCRIPT=03_codigo/CIC17/04_training/$NOMBRE_SCRIPT

BASE=/home/inginf/u32902122/TFG/$PATH_SCRIPT
NOTEBOOK=$BASE.ipynb
NOTEBOOK_OUT="/home/inginf/u32902122/TFG/04_experimentos/logs/notebooks/${NOMBRE_SCRIPT}_executed_${SLURM_JOB_ID}.ipynb"

CSV_NVIDIA="/home/inginf/u32902122/TFG/04_experimentos/logs/nvidia/${NOMBRE_SCRIPT}_${SLURM_JOB_ID}_nvidia-smi.csv"
TXT_NVIDIA="/home/inginf/u32902122/TFG/04_experimentos/logs/nvidia/${NOMBRE_SCRIPT}_${SLURM_JOB_ID}_nvidia-smi.txt"

module load Anaconda3/2024.02-1
source activate tfg

#------- Comando -------
nvidia-smi > $TXT_NVIDIA
cd $SLURM_SUBMIT_DIR

nvidia-smi -lms 1000 --query-gpu=timestamp,pstate,power.management,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,temperature.gpu,temperature.memory,memory.used,memory.total,memory.free,clocks.current.sm,clocks.current.memory --format=csv,nounits -f "$CSV_NVIDIA" &

conda activate tfg

# Comando que realiza el trabajo:
jupyter nbconvert \
  --execute \
  --to notebook \
  --ExecutePreprocessor.kernel_name=tfg \
  --ExecutePreprocessor.timeout=86400 \
  --output "${NOTEBOOK_OUT}" \
  "${NOTEBOOK}"