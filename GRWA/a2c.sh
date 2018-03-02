#!/usr/bin/env bash

# python version should be 3.x
python -u main.py --mode learning \
--net USNET.md --wave-num 10 --rou 10 --miu 1000 --max-iter 3000 \
--k 1 --weight None --workers 4 --steps 3000000 \
--img-height 224 --img-width 224


