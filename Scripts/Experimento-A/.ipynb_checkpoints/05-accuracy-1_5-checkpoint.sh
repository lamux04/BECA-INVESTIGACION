#!/bin/bash
#SBATCH --job-name=05-accuracy-1_5
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=200GB
#SBATCH --output=/home/inginf/u32902122/TFG/Scripts/Experimento-A/05-accuracy-1_5-output.log

#SBATCH --mail-user=javier.labradormunoz@alum.uca.es
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

NOMBRE_PASO="05-accuracy-1_5"
BASE=/home/inginf/u32902122/TFG/Scripts/Experimento-A/$NOMBRE_PASO
NOTEBOOK=$BASE.ipynb
CSV_NVIDIA=$BASE-nvidia-smi.csv
OUT_NVIDIA=$BASE-nvidia-smi.output
LOG=$BASE-output.log

module load Anaconda3/2024.02-1

#------- Comando -------
cd $SLURM_SUBMIT_DIR
# Comando que realiza el trabajo:
jupyter nbconvert --execute --to notebook --ExecutePreprocessor.kernel_name=t5env $NOTEBOOK