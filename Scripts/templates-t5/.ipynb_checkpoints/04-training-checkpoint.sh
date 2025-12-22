#!/bin/bash
#SBATCH --job-name=NUMBER-CODE-VERSION-FOLD
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --time=NDAYS-00:00:00
#SBATCH --mem=200GB
#SBATCH --output=/home/inginf/u48977630/MAINDIR/NUMBER-CODE-NAME/NUMBER-CODE-VERSION-FOLD-NAME-MODEL-1-training-output.log

#SBATCH --mail-user=leopoldo.gutierrez@uca.es
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

module load Anaconda3/2022.05
module load Python/3.9.6-GCCcore-11.2.0

#------- Comando -------
nvidia-smi > /home/inginf/u48977630/MAINDIR/NUMBER-CODE-NAME/NUMBER-CODE-VERSION-FOLD-NAME-MODEL-1-training-nvidia-smi.output
cd $SLURM_SUBMIT_DIR

nvidia-smi -lms 1000 --query-gpu=timestamp,pstate,power.management,power.draw,power.limit,power.default_limit,power.min_limit,power.max_limit,temperature.gpu,temperature.memory,memory.used,memory.total,memory.free,clocks.current.sm,clocks.current.memory --format=csv,nounits -f "/home/inginf/u48977630/MAINDIR/NUMBER-CODE-NAME/NUMBER-CODE-VERSION-FOLD-NAME-MODEL-1-training-nvidia-smi.csv" &

# Comando que realiza el trabajo:
jupyter nbconvert --execute --to notebook /home/inginf/u48977630/MAINDIR/NUMBER-CODE-NAME/NUMBER-CODE-VERSION-FOLD-NAME-MODEL-1-training.ipynb