# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import sem


def xcorr(X, Y, grouping=None):
    """
    Calculates the cross-covariance matrix of ``X`` and ``Y``

    Parameters
    ----------
    X : (N x J) array_like
    Y : (N x K) array_like
    grouping : (N,) array_like, optional
        Grouping array, where ``len(np.unique(grouping))`` is the number of
        distinct groups in ``X`` and ``Y``. Cross-covariance matrices are
        computed separately for each group and are stacked row-wise.

    Returns
    -------
    (K[*G] x J) np.ndarray
        Cross-covariance of ``X`` and ``Y``
    """

    if grouping is None:
        return _compute_xcorr(X, Y)
    else:
        return np.row_stack([_compute_xcorr(X[grouping == grp],
                                            Y[grouping == grp])
                             for grp in np.unique(grouping)])


def _compute_xcorr(X, Y):
    """
    Calculates the cross-covariance matrix of ``X`` and ``Y``

    Parameters
    ----------
    X : (N x J) array_like
    Y : (N x K) array_like

    Returns
    -------
    xprod : (K x J) np.ndarray
        Cross-covariance of ``X`` and ``Y``
    """

    Xnz, Ynz = normalize(zscore(X)), normalize(zscore(Y))
    xprod = (Ynz.T @ Xnz) / (Xnz.shape[0] - 1)

    return xprod


def zscore(X):
    """
    Z-scores ``X`` by subtracting mean and dividing by standard deviation

    Effectively the same as ``np.nan_to_num(scipy.stats.zscore(X))`` but
    handles DivideByZero without issuing annoying warnings.

    Parameters
    ----------
    X : (N x J) array_like

    Returns
    -------
    zarr : (N x J) np.ndarray
        Z-scored ``X``
    """

    arr = np.array(X)

    avg, stdev = arr.mean(axis=0), arr.std(axis=0)
    zero_items = np.where(stdev == 0)[0]

    if zero_items.size > 0:
        avg[zero_items], stdev[zero_items] = 0, 1

    zarr = (arr - avg) / stdev
    zarr[:, zero_items] = 0

    return zarr


def normalize(X, axis=0):
    """
    Normalizes ``X`` along ``axis``

    Utilizes Frobenius norm (or Hilbert-Schmidt norm, L_{p,q} norm where p=q=2)

    Parameters
    ----------
    X : (N x K) array_like
        Data to be normalized
    axis : int, optional
        Axis for normalization. Default: 0

    Returns
    -------
    normed : (N x K) np.ndarray
        Normalized ``X``
    """

    normed = np.array(X)
    normal_base = np.linalg.norm(normed, axis=axis, keepdims=True)

    # avoid DivideByZero errors
    zero_items = np.where(normal_base == 0)
    normal_base[zero_items] = 1
    # normalize and re-set zero_items to 0
    normed = normed / normal_base
    normed[zero_items] = 0

    return normed


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

    U_rel = U_orig / sem(U_boot, axis=-1)
    V_rel = V_orig / sem(V_boot, axis=-1)

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


def get_seed(seed=None):
    """
    Determines type of ``seed`` and returns RandomState instance

    Parameters
    ----------
    seed : {int, RandomState instance, None}, optional
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, ``seed`` is the seed used by the random number
        generator. If RandomState instance, ``seed`` is the random number
        generator. If None, the random number generator is the RandomState
        instance used by ``np.random``. Default: None

    Returns
    -------
    RandomState instance
    """

    if seed is not None:
        if isinstance(seed, int):
            return np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            return seed
    return np.random


def dummy_code(grouping):
    """
    Dummy codes ``grouping``

    Parameters
    ----------
    grouping : (N,) array_like
        Array with labels separating ``N`` subjects into ``G`` groups

    Returns
    -------
    Y : (N x G) np.ndarray
        Dummy coded grouping array
    """

    groups = np.unique(grouping)
    Y = np.column_stack([(grouping==grp).astype(int) for grp in groups])

    return Y
