#!/usr/bin/env bash

# python version should be 3.x
python -u ../main.py --mode learning --cnn simplenet --num-steps 8 \
--net 6node.md --wave-num 10 --rou 8 --miu 300 --max-iter 1000 \
--k 1 --weight None --workers 16 --steps 10000000 \
--img-height 224 --img-width 224


