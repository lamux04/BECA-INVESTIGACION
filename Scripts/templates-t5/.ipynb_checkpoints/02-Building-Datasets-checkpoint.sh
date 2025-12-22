#!/bin/bash
#SBATCH --job-name=02L6
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --time=0-1:00:00
#SBATCH --mem=200GB
#SBATCH --output=/home/inginf/u48977630/experimentos_v2/02L6-Building-Datasets-output.log

#SBATCH --mail-user=leopoldo.gutierrez@uca.es
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

module load Anaconda3/2022.05
module load Python/3.9.6-GCCcore-11.2.0

cd $SLURM_SUBMIT_DIR
# Comando que realiza el trabajo:
mkdir -p 02-built-dataset
jupyter nbconvert --execute --to notebook /home/inginf/u48977630/experimentos_v2/02L6-Building-Datasets.ipynb