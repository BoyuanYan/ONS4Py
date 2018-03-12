#!/usr/bin/env sh

python -u ../main.py --mode learning --cnn simplenet --num-steps 2 --step-over one_service -e \
--net 6node.md --wave-num 5 --rou 8 --miu 300 --max-iter 3000 \
--k 1 --weight None --workers 4 --steps 3000 \
--img-height 112 --img-width 112


