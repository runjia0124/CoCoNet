#!/bin/bash 
start_time=$(date +%s)  

python main.py --train --c1 0.5 --c2 0.75 --epoch 30 --bs 30 \
               --logdir ./logs_Jan1423 --use_gpu

python main.py --finetune --c1 0.5 --c2 0.75 --epoch 2 --bs 30 \
               --logdir ./logs_Jan1423 --use_gpu

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
min=$(expr $cost_time / 60)
sec=$(expr $cost_time % 60)
echo "Training time is $min min $sec s"

