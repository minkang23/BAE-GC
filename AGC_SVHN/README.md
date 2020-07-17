## Autoencoder-based Graph Construction for Semi-supervised Learning

Tensorflow implementation for reproducing the Semi-supervised learning results on SVHN (Street Veiw House Number) dataset in the paper (CVPR2020 #10225).



### Requirements

Python 2.7, Tensorflow-gpu == 1.14.0, Numpy, Scipy



### Training

**SVHN with 1000 labels**

```CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'svhn' --whiten_norm 'norm' --augment_mirror False --augment_translation 2 --n_labeled 1000 --lr_max 0.003 --ratio_max 50.0 -e 200 --random_seed 0 --dims 500 100 10 --coef_emb 0.2```



**SVHN with 500 labels**

```CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'svhn' --whiten_norm 'norm' --augment_mirror False --augment_translation 2 --n_labeled 500 --lr_max 0.003 --ratio_max 50.0 -e 200 --random_seed 0 --dims 500 100 10 --coef_emb 0.2```



**SVHN with 250 labels**

```CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'svhn' --whiten_norm 'norm' --augment_mirror False --augment_translation 2 --n_labeled 250 --lr_max 0.003 --ratio_max 50.0 -e 200 --random_seed ${seed} --dims 500 100 10 --coef_emb 0.8```



### Run with 10 different random seed

```CUDA_VISIBLE_DEVICES=0 sh run.sh```