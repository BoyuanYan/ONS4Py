#!/usr/bin/env sh


#SBATCH --partition=Test
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -n1
#SBATCH --job-name=a2c
#SBATCH -o log-test-%j

srun python -u ../main.py --mode learning --cnn simplenet --num-steps 16 \
--net 6node.md --wave-num 10 --rou 8 --miu 300 --max-iter 300 \
--k 1 --weight None --workers 16 --steps 10e6 \
--img-height 224 --img-width 224


