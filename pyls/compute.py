# -*- coding: utf-8 -*-

import numpy as np


def perm_sig(orig, perm):
    """
    Calculates significance of ``orig`` values by permutation

    Compares amplitude of each singular value to distribution created via
    permutation in ``perm``

    Parameters
    ----------
    orig : (L x L) array_like
        Diagonal matrix of singular values
    perm : (L x P) array_like
        Distribution of singular values from permutation testing where ```P``
        is the number of permutations

    Returns
    -------
    pvals : (L,) np.ndarray
        P-values of singular values
    """

    pvals = np.sum(perm > np.diag(orig)[:, None], axis=1) / perm.shape[-1]

    return pvals


def boot_ci(boot, ci=95):
    """
    Generates CI for bootstrapped values ``boot``

    Parameters
    ----------
    boot : (K x L x B) array_like
        Singular vectors, where ``K`` is variables, ``L`` is components, and
        ``B`` is bootstraps
    ci : (0, 100) float, optional
        Confidence interval bounds to be calculated. Default: 95

    Returns
    -------
    lower : (K x L) np.ndarray
        Lower bound of CI for singular vectors in ``boot``
    upper : (K x L) np.ndarray
        Upper bound of CI for singular vectors in ``boot``
    """

    low = (100 - ci) / 2
    prc = [low, 100 - low]

    lower, upper = np.percentile(boot, prc, axis=-1)

    return lower, upper


def boot_rel(orig, boot):
    """
    Determines bootstrap ratios (BSR) of saliences from bootstrap distributions

    Parameters
    ----------
    orig : (K x L) array_like
        Original singular vectors
    boot : (K x L x B) array_like
        Bootstraped singular vectors, where ``B`` is bootstraps

    Returns
    -------
    bsr : (K[*G] x L) ndarray
        Bootstrap ratios for provided singular vectors
    """

    bsr = orig / boot.std(axis=-1, ddof=1)

    return bsr


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

    return crossblock_cov(singular) > (1 / (len(singular) + 1))


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
    resamp : np.ndarray
        Singular values of rotated ``permuted`` matrix
    rotate : np.ndarray
        Matrix for rotating ``permuted`` to ``original``
    """

    N, _, P = np.linalg.svd(original.T @ permuted)
    rotate = N @ P
    resamp = permuted @ singular @ rotate

    return resamp, rotate


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

    X_mean_centered = X - get_group_mean(X, Y)

    return X_mean_centered
