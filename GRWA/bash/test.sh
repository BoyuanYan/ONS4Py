#!/usr/bin/env sh

python -u ../main.py --mode learning --cnn simplenet --num-steps 2 --step-over one_service \
--net 6node.md --wave-num 5 --rou 8 --miu 300 --max-iter 300 \
--k 1 --weight None --workers 1 --steps 10e6 \
--img-height 224 --img-width 224


