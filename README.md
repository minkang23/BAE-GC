# BAE-GC
## Autoencoder-based Graph Construction for Semi-supervised Learning (ECCV 2020 #4655).

Tensorflow implementation for reproducing the Semi-supervised learning results on MNIST,SVHN and CIFAR-10 datasetS in the paper



### Requirements

Python 2.7, Tensorflow-gpu == 1.14.0, Numpy, Scipy



### Training

**SVHN with 1000 labels**

```CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'svhn' --whiten_norm 'norm' --augment_mirror False --augment_translation 2 --n_labeled 1000 --lr_max 0.003 --ratio_max 50.0 -e 200 --random_seed 0 --dims 500 100 10 --coef_emb 0.2```



**SVHN with 500 labels**

```CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'svhn' --whiten_norm 'norm' --augment_mirror False --augment_translation 2 --n_labeled 500 --lr_max 0.003 --ratio_max 50.0 -e 200 --random_seed 0 --dims 500 100 10 --coef_emb 0.2```



**SVHN with 250 labels**

```CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'svhn' --whiten_norm 'norm' --augment_mirror False --augment_translation 2 --n_labeled 250 --lr_max 0.003 --ratio_max 50.0 -e 200 --random_seed 0 --dims 500 100 10 --coef_emb 0.8```


**CIFAR-10: 4000 labels**

```CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar-10' --whiten_norm 'zca' --augment_mirror True --augment_translation 2 --n_labeled 4000 --lr_max 0.1 --ratio_max 100.0 -e 400 --random_seed 0 --dims 500 100 10 --margin 0.5 --coeff 0.2```

**CIFAR-10: 2000 labels**

```CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar-10' --whiten_norm 'zca' --augment_mirror True --augment_translation 2 --n_labeled 2000 --lr_max 0.1 --ratio_max 100.0 -e 400 --random_seed 0 --dims 500 100 10 --margin 0.5 --coeff 0.2```


**CIFAR-10: 1000 labels**

```CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar-10' --whiten_norm 'zca' --augment_mirror True --augment_translation 2 --n_labeled 1000 --lr_max 0.1 --ratio_max 10.0 -e 400 --random_seed 0 --dims 500 100 10 --margin 0.5 --coeff 0.2 --mixup_sup_alpha 0.2 --mixup_usup_alpha 0.2```
