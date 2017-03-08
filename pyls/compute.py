#!/usr/bin/env python

import multiprocessing as mp
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.stats import zscore, sem


def svd(X, Y, k, norm=True):
    """
    Runs SVD on the covariance matrix of `X` and `Y`

    Uses sklearn.utils.extmath.randomized_svd for computation of a truncated
    SVD.

    Parameters
    ----------
    X : array (N x j)
    Y : array (N x k)
    k : int
        rank of Y.T @ X matrix; determines # of singular vectors generated
    norm : bool
        whether to zscore X and Y prior to SVD

    Returns
    -------
    U, d, V : left singular vectors, singular values, right singular vectors
    """

    if norm: X, Y = zscore(X), zscore(Y)
    U, d, V = randomized_svd(Y.T @ X, n_components=k)

    return U, np.diag(d), V.T


def procrustes(permuted, original, singular):
    """
    Performs Procrustes rotation on `permuted` to align with `orig`

    Parameters
    ----------
    permuted : array
    original : array
    singular : array

    Returns
    -------
    array : singular values of rotated `permuted` matrix
    """

    N, O, P = np.linalg.svd(original.T @ permuted)
    Q = N @ P
    resamp = permuted @ singular @ Q

    return resamp, Q


def permute(X, Y, k, original, perms=1000, procs=1):
    """
    Permutes `X` (w/o replacement) and recomputes singular values

    Uses procrustes rotation to ensure SVD is in same space as `original`

    Parameters
    ----------
    X : array (N x j)
    Y : array (N x t)
    k : int
        rank of Y.T @ X matrix
    original : array
    perms : int
        number of permutations to run
    procs : int
        function will multiprocess permutations for potential speed-up

    Returns
    -------
    array : distributions of singular values
    """

    permuted_values = np.zeros((perms,k))

    for i in range(perms):
        X_perm = np.random.permutation(X)
        U, d, V = svd(X_perm, Y, k, norm=False)

        if len(U) < len(V): permuted = U
        else: permuted = V

        resamp, *rest = procrustes(permuted, original, d)
        permuted_values[i] = np.sqrt((resamp**2).sum(axis=0))

    return permuted_values


def bootstrap(X, Y, k, U_orig, V_orig, boots=500, procs=1):
    """
    Bootstrap `X`/`Y` (with replacement) and computes SE of singular values

    Parameters
    ----------
    X : array (N x j)
    Y : array (N x t)
    k : int
        rank of Y.T @ X matrix
    Uorig : array (t x k)
    Vorig : array (j x k)
    boots : int
        number of boostraps to run
    procs : int
        function will multiprocess bootstraps for potential speed-up

    Returns
    -------
    array, array : boostrapped left, bootstrapped right singular vectors
    """

    U_boot = np.zeros(U_orig.shape + (boots,))
    V_boot = np.zeros(V_orig.shape + (boots,))
    I = np.identity(k)

    for i in range(boots):
        inds = np.random.choice(np.arange(len(X)),size=len(X),replace=True)
        X_boot, Y_boot = X[inds], Y[inds]
        U, d, V = svd(X_boot, Y_boot, k, norm=True)

        U_boot[:,:,i], Q = procrustes(U, U_orig, I)
        V_boot[:,:,i] = V @ Q

    return U_boot, V_boot


def perm_sig(permuted_values, orig_values):
    """
    Calculates significance of `orig_values` by comparing amplitude of each
    to distribution in `permuted_values`

    Parameters
    ----------
    permuted_values : array (n_perms x k)
        distribution of singular values from permutation testing
    orig_values : array (diagonal, k x k)
        singular values from original SVD

    Returns
    -------
    array : p-values of singular values from original SVD
    """

    pvals = np.zeros(len(orig_values))
    perms = len(permuted_values.shape)

    for i in range(len(pvals)):
        top_of_dist = np.where(permuted_values[:,i]>orig_values[i,i])[0]
        pvals[i] = top_of_dist.size/perms

    return pvals


def boot_ci(U_boot, V_boot, p=.01):
    """
    Generates CI for bootstrapped values

    Parameters
    ----------
    U_boot : array (t x k x boot)
    V_boot : array (j x k x boot)
    p : float (0,1)
        determines bounds of CI

    Returns
    -------
    array, array : CI for U (t x k x 2), CI for V (j x k x 2)
    """
    low = 100*(p/2)
    high = 100-low

    U_out = np.zeros(U_boot.shape[0:2]+(2,))
    V_out = np.zeros(V_boot.shape[0:2]+(2,))

    for n, f in enumerate([low,high]):
        U_out[:,:,n] = np.percentile(U_boot,f,axis=2)
        V_out[:,:,n] = np.percentile(V_boot,f,axis=2)

    return U_out, V_out


def boot_rel(U_orig, V_orig, U_boot, V_boot):
    """
    Determines bootstrap ratios of saliences from bootstrapped distributions

    Parameters
    ----------
    U_orig : array (t x k)
    V_orig : array (j x k)
    U_boot : array (t x k x boot)
    V_boot : array (j x k x boot)

    Returns
    -------
    array, array : BSR for U (t x k), BSR for V (j x k)
    """

    U_rel = U_orig/sem(U_boot,axis=2)
    V_rel = V_orig/sem(V_boot,axis=2)

    return U_rel, V_rel
