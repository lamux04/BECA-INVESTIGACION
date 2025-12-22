#!/bin/bash
#SBATCH --job-name=01-Preprocessing-Cleaning.ipynb
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --time=0-6:00:00
#SBATCH --mem=120GB
#SBATCH --output=/home/inginf/u48977630/experimentos_v2/01-Preprocessing-Cleaning-output.log

#SBATCH --mail-user=leopoldo.gutierrez@uca.es
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

module load Anaconda3/2022.05
module load Python/3.9.6-GCCcore-11.2.0

#------- Comando -------
cd $SLURM_SUBMIT_DIR
# Comando que realiza el trabajo:
find ../dataset/ -type f -exec sed -i '1p;/Protocol/d' {} \;
mkdir -p 01-clean-dataset
mkdir -p 01-clean-dataset/01-clean-dataset-dup
mkdir -p 01-clean-dataset/01-clean-dataset-no-dup
jupyter nbconvert --execute --to notebook /home/inginf/u48977630/experimentos_v2/01-Preprocessing-Cleaning.ipynb