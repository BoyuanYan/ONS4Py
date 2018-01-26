#!/usr/bin/env sh

#SBATCH --partition=Liveness
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH -n8
#SBATCH --job-name=resnet18
#SBATCH -o log-resnet18-%j

srun python -u main_my.py -a resnet18 --save-path resnet18
