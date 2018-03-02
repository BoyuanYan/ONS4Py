#!/usr/bin/env bash

# python version should be 3.x
python -u main.py --mode alg \
--net 6node.md --wave-num 80 --rou 10 --miu 3000 --max-iter 10000 \
--k 1 --weight None --workers 4 --steps 300000

