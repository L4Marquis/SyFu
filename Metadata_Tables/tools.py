# Code by Landry Marquis based on https://github.com/cguilloteau/Fast-fusion-of-astronomical-images 
#
# If you use this code, please cite the paper: 
# Marquis L. et al., Fusion of JWST data: Demonstrating practical feasibility,
# Astronomy & Astrophysics, volume 708, 2026

import numpy as np


def subsample_list_uniformly(list, subsample_factor):
    return list[0::subsample_factor]

def overspl_by_copy(list, oversampling_factor):
    new_list = []
    for i in range(len(list)):
        for j in range(oversampling_factor):
            new_list.append([list[i]])
    return new_list

def aliasing(X, shape, ratio):
    l, m, n = shape
    o = m // ratio
    p = n // ratio
    X_aliasing = np.zeros((l, o, p), complex)
    X = np.reshape(X, shape)
    for i in range(ratio):
        for j in range(ratio):
            X_aliasing += X[:, i * o : (i+1) * o, j * p : (j+1) * p]
    return np.reshape(X_aliasing, (l, o * p)) / ratio

def aliasing_adjoint(X, shape, ratio):
    M = shape[1] // ratio
    N = shape[2] // ratio
    X_ = np.zeros(shape, complex)
    X = np.reshape(X, (X.shape[0], M, N))
    for i in range(ratio):
        for j in range(ratio):
            X_[:, i * M : (i+1) * M, j * N : (j+1) * N] = X
    return np.reshape(X_ / ratio, (X_.shape[0], X_.shape[1] * X_.shape[2]))
