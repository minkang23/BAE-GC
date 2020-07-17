#!/usr/bin/env bash
# SVHN: 1000 labels
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#python main.py --dataset 'svhn' --whiten_norm 'norm' --augment_mirror False --augment_translation 2 --n_labeled 1000 --lr_max 0.003 --ratio_max 50.0 -e 200 --random_seed ${seed} --dims 500 100 10 --coef_emb 0.2
#done
# SVHN: 500 labels
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#python main.py --dataset 'svhn' --whiten_norm 'norm' --augment_mirror False --augment_translation 2 --n_labeled 500 --lr_max 0.003 --ratio_max 50.0 -e 200 --random_seed ${seed} --dims 500 100 10 --coef_emb 0.2
#done
# SVHN: 250 labels
for seed in 0 1 2 3 4 5 6 7 8 9
do
python main.py --dataset 'svhn' --whiten_norm 'norm' --augment_mirror False --augment_translation 2 --n_labeled 250 --lr_max 0.003 --ratio_max 50.0 -e 200 --random_seed ${seed} --dims 500 100 10 --coef_emb 0.8
done