#!/usr/bin/env bash

# python version should be 3.x
python -u main.py --mode alg \
--net 6node.md --wave-num 10 --rou 8 --miu 300 --max-iter 3000 \
--k 1 --weight None --workers 4 --steps 30000

