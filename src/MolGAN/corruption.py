#corruption.py
import numpy as np

def apply_masking(X, mask_ratio=0.1):
    X = X.copy()
    num_elements = np.prod(X.shape)
    num_mask = int(mask_ratio * num_elements)
    indices = np.unravel_index(np.random.choice(num_elements, num_mask, replace=False), X.shape)
    X[indices] = 0
    return X

def apply_deshuffling(X):
    X = X.copy()
    for i in range(X.shape[0]):
        perm = np.random.permutation(X.shape[1])
        X[i] = X[i][perm]
    return X

def apply_noise_injection(X, noise_level=0.05):
    X = X.copy()
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise
