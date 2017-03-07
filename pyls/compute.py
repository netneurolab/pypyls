#!/usr/bin/env python

import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.stats import zscore


def svd(X, Y, norm=True):
    """
    Runs SVD on the covariance matrix of `X` and `Y`

    Parameters
    ----------
    X : array (N x j)
    Y : array (N x k)
    norm : bool
        Whether to zscore X and Y prior to singular value decomposition

    Returns
    -------
    U, d, V : left singular vectors, singular values, right singular vectors
    """

    if norm: X, Y = zscore(X), zscore(Y)

    k = min(min(Y.shape), min(X.shape))  # lowest rank of input matrices
    U, d, V = randomized_svd(Y.T @ X, n_components=k)

    return U, d, V.T
