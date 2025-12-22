#!/bin/bash
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu
#SBATCH --time=0-15:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=comprobar-datos
#SBATCH --output=/home/inginf/u32902122/TFG/Scripts/Experimento-A/comprobar-datos-output.log
#SBATCH --mail-user=javier.labradormunoz@alum.uca.es
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

NOMBRE_PASO=comprobar-datos
BASE=/home/inginf/u32902122/TFG/Scripts/Experimento-A/$NOMBRE_PASO
NOTEBOOK=$BASE.ipynb
CSV_NVIDIA=$BASE-nvidia-smi.csv
OUT_NVIDIA=$BASE-nvidia-smi.output
LOG=$BASE-output.log



module load Anaconda3/2024.02-1

#------- Comando -------
nvidia-smi > $OUT_NVIDIA
cd $SLURM_SUBMIT_DIR

nvidia-smi -lms 1000 --query-gpu=timestamp,pstate,power.management,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,temperature.gpu,temperature.memory,memory.used,memory.total,memory.free,clocks.current.sm,clocks.current.memory --format=csv,nounits -f "$CSV_NVIDIA" &

conda activate probandot5

# Comando que realiza el trabajo:
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.kernel_name=python3 $NOTEBOOK