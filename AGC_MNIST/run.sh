#!/usr/bin/env bash
# MNIST: Pi
## 100 labels
for seed in 1
do
python main.py --dataset 'mnist' --whiten_norm 'norm' --augment_mirror False --augment_translation 0 --n_labeled 100 --lr_max 0.0001 --ratio_max 100.0 -e 300 --random_seed ${seed} --dims 300 300 15 --coef_emb 0.4
done

### 50 labels
#for seed in 1
#do
#python main.py --dataset 'mnist' --whiten_norm 'norm' --augment_mirror False --augment_translation 0 --n_labeled 50 --lr_max 0.0001 --ratio_max 100.0 -e 300 --random_seed ${seed} --dims 300 300 15 --coef_emb 0.4
#done

### 20 labels
#for seed in 1
#do
#python main.py --dataset 'mnist' --whiten_norm 'norm' --augment_mirror False --augment_translation 0 --n_labeled 20 --lr_max 0.0001 --ratio_max 100.0 -e 300 --random_seed ${seed} --dims 300 300 15 --coef_emb 0.4
#done
