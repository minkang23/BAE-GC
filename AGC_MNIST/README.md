## Autoencoder-based Graph Construction for Semi-supervised Learning

Tensorflow implementation for reproducing the Semi-supervised learning results on MNIST dataset in the paper (CVPR2020 #10225).



### Requirements

Python 2.7, Tensorflow-gpu == 1.14.0, Numpy, Scipy



### Training

**MNIST with 100 labels**

```python main.py --dataset 'mnist' --whiten_norm 'norm' --augment_mirror False --augment_translation 0 --n_labeled 100  --lr_max 0.0001 --ratio_max 100.0 -e 300 --random_seed 1 --dims 300 300 15 --coef_emb 0.4```



**MNIST with 50 labels**

```python main.py --dataset 'mnist' --whiten_norm 'norm' --augment_mirror False --augment_translation 0 --n_labeled 50  --lr_max 0.0001 --ratio_max 100.0 -e 300 --random_seed 1 --dims 300 300 15 --coef_emb 0.4```



**MNIST with 20 labels**

```python main.py --dataset 'mnist' --whiten_norm 'norm' --augment_mirror False --augment_translation 0 --n_labeled 20  --lr_max 0.0001 --ratio_max 100.0 -e 300 --random_seed 1 --dims 300 300 15 --coef_emb 0.4````

