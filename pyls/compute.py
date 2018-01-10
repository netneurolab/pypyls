# -*- coding: utf-8 -*-

import numpy as np


def perm_sig(permuted_singular, orig_singular):
    """
    Calculates significance of ``orig_singular`` values

    Compares amplitude of each singular value to distribution created via
    permutation in ``permuted_singular``

    Parameters
    ----------
    permuted_singular : (P x L) array_like
        Distribution of singular values from permutation testing where ``P``
        is the number of permutations and ``L`` is the number of components
        from the SVD
    orig_singular : (L x L) array_like
        Diagonal matrix of singular values from original SVD

    Returns
    -------
    pvals : (L,) np.ndarray
        P-values of singular values from original SVD
    """

    pvals = np.zeros(len(orig_singular))
    n_perm = len(permuted_singular)

    for i in range(len(pvals)):
        top_dist = np.argwhere(permuted_singular[:, i] > orig_singular[i, i])
        pvals[i] = top_dist.size / n_perm

    return pvals


def boot_ci(U_boot, V_boot, ci=95):
    """
    Generates CI for bootstrapped values ``U_boot`` and ``V_boot``

    Parameters
    ----------
    U_boot : (K[*G] x L x B) array_like
    V_boot : (J x L x B) array_like
    ci : (0, 100) float, optional
        Confidence interval bounds to be calculated. Default: 95

    Returns
    -------
    (K[*G] x L x 2) ndarray
        Bounds of confidence interval for left singular vectors
    (J x L x 2) array
        Bounds of confidence interval for right singular vectors
    """

    low = (100 - ci) / 2
    prc = [low, 100 - low]

    U_ci = np.percentile(U_boot, prc, axis=2).transpose(1, 2, 0)
    V_ci = np.percentile(V_boot, prc, axis=2).transpose(1, 2, 0)

    return U_ci, V_ci


def boot_rel(U_orig, V_orig, U_boot, V_boot):
    """
    Determines bootstrap ratios (BSR) of saliences from bootstrap distributions

    Parameters
    ----------
    U_orig : (K[*G] x L) array_like
    V_orig : (J x L) array_like
    U_boot : (K[*G] x L x B) array_like
    V_boot : (J x L x B) array_like

    Returns
    -------
    (K[*G] x L) ndarray
        Bootstrap ratios for left singular vectors
    (J x L) ndarray
        Bootstrap ratios for right singular vectors
    """

    U_rel = U_orig / U_boot.std(axis=-1, ddof=1)
    V_rel = V_orig / V_boot.std(axis=-1, ddof=1)

    return U_rel, V_rel


def crossblock_cov(singular):
    """
    Calculates cross-block covariance of ``singular`` values

    Cross-block covariances details amount of variance explained

    Parameters
    ----------
    singular : (L x L) array_like
        Diagonal matrix of singular values

    Returns
    -------
    (L,) np.ndarray
        Cross-block covariance
    """

    squared_sing = np.diag(singular)**2

    return squared_sing / squared_sing.sum()


def kaiser_criterion(singular):
    """
    Determines if variance explained by ``singular`` value > Kaiser criterion

    Kaiser criterion is 1/# singular values. If cross-block covariance
    explained by singular value exceeds criterion, return True; else, return
    False.

    Parameters
    ----------
    singular : (L x L) array_like
        Diagonal matrix of singular values from original SVD

    Returns
    -------
    (L,) np.ndarray
        Boolean array detailing whether singular value passes Kaiser criterion
    """

    return crossblock_cov(singular) > (1 / len(singular))


def boot_sig(boot):
    """
    Determines which entries of ``boot`` are significant via CI crossing

    If CI crosses zero, then bootstrap value is not

    Parameters
    ----------
    boot : (F x L x 2) array_like
        One of the outputs of ``boot_ci()``

    Returns
    -------
    (F,) ndarray
        Boolean array
    """

    return np.sign(boot).sum(axis=-1).astype('bool')


def procrustes(original, permuted, singular):
    """
    Performs Procrustes rotation on ``permuted`` to align with ``original``

    ``original`` and ``permuted`` should be either left *or* right singular
    vector from two SVDs. ``singular`` should be the diagonal matrix of
    singular values from the SVD that generated ``original``

    Parameters
    ----------
    original : array_like
    permuted : array_like
    singular : array_like

    Returns
    -------
    ndarray
        Singular values of rotated ``permuted`` matrix
    """

    N, _, P = np.linalg.svd(original.T @ permuted)
    Q = N @ P
    resamp = permuted @ singular @ Q

    return resamp, Q


def get_group_mean(X, Y, grand=True):
    """
    Parameters
    ----------
    X : (N x K) array_like
    Y : (N x G) array_like
        Dummy coded group array
    grand : bool, optional
        Default : True

    Returns
    -------
    group_mean : {(G,) or (G x K)} np.ndarray
        If grand is set, returns array with shape (G,); else, returns (G x K)
    """

    group_mean = np.zeros((Y.shape[-1], X.shape[-1]))

    for n, grp in enumerate(Y.T.astype('bool')):
        group_mean[n] = X[grp].mean(axis=0)

    if grand:
        return group_mean.sum(axis=0) / Y.shape[-1]
    else:
        return group_mean


def get_mean_norm(X, Y):
    """
    Parameters
    ----------
    X : (N x K) array_like
    Y : (N x G) array_like
        Dummy coded group array

    Returns
    -------
    X_mean_centered : (N x K) np.ndarray
        ``X`` centered based on grand mean (i.e., mean of group means)
    """

    grand_mean = get_group_mean(X, Y)
    X_mean_centered = np.zeros_like(X)

    for grp in Y.T.astype('bool'):
        X_mean_centered[grp] = X[grp] - grand_mean

    return X_mean_centered
