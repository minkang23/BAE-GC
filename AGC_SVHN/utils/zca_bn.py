import numpy as np
from scipy import linalg

class ZCA(object):
    def __init__(self, epsilon=1e-5, x=None):
        s = x.shape
        x = x.copy().reshape((s[0], np.prod(s[1:])))
        self.m = np.mean(x, axis=0)
        x -= self.m

        sigma = np.dot(x.T, x) / x.shape[0]
        U, S, V = linalg.svd(sigma)

        self.ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(S + epsilon))), U.T)

    def apply(self, x):
        s = x.shape
        x = x.copy().reshape((s[0], np.prod(s[1:])))

        x -= self.m

        return np.transpose(np.dot(x, self.ZCAMatrix).reshape(s), [0, 2, 3, 1])