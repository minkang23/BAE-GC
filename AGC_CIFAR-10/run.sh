#!/usr/bin/env bash

## CIFAR-10: 4000 labels
#for seed in 0 1
#do
#python main.py --dataset 'cifar-10' --whiten_norm 'zca' --augment_mirror True --augment_translation 2 --n_labeled 4000 --lr_max 0.1 --ratio_max 100.0 -e 400 --random_seed ${seed} --dims 500 100 10 --margin 0.5 --coeff 0.2
#done
#
## CIFAR-10: 2000 labels
#for seed in 0 1
#do
#python main.py --dataset 'cifar-10' --whiten_norm 'zca' --augment_mirror True --augment_translation 2 --n_labeled 2000 --lr_max 0.1 --ratio_max 100.0 -e 400 --random_seed ${seed} --dims 500 100 10 --margin 0.5 --coeff 0.2
#done

## CIFAR-10: 1000 labels
for seed in 0 1
do
python main.py --dataset 'cifar-10' --whiten_norm 'zca' --augment_mirror True --augment_translation 2 --n_labeled 1000 --lr_max 0.1 --ratio_max 10.0 -e 400 --random_seed ${seed} --dims 500 100 10 --margin 0.5 --coeff 0.2 --mixup_sup_alpha 0.2 --mixup_usup_alpha 0.2
done