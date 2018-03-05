#!/usr/bin/env sh


#SBATCH --partition=V100C16
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -n1
#SBATCH --job-name=a2c
#SBATCH -o test-a2c-simplenet-service-%j

srun python -u ../main.py --mode learning --cnn simplenet --step-over one_time --evaluate \
--net 6node.md --wave-num 10 --rou 8 --miu 300 --max-iter 30 \
--k 1 --weight None \
--img-height 224 --img-width 224


