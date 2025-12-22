#!/bin/bash
#SBATCH --job-name=NUMBER-CODE-VERSION-FOLD-Ac
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --time=NDAYS-00:00:00
#SBATCH --mem=200GB
#SBATCH --output=/home/inginf/u48977630/MAINDIR/NUMBER-CODE-NAME/NUMBER-CODE-VERSION-FOLD-NAME-MODEL-2-accuracy-output.log

#SBATCH --mail-user=leopoldo.gutierrez@uca.es
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

module load Anaconda3/2022.05
module load Python/3.9.6-GCCcore-11.2.0

#------- Comando -------
cd $SLURM_SUBMIT_DIR
# Comando que realiza el trabajo:
jupyter nbconvert --execute --to notebook /home/inginf/u48977630/MAINDIR/NUMBER-CODE-NAME/NUMBER-CODE-VERSION-FOLD-NAME-MODEL-2-accuracy.ipynb