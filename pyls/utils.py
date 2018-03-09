# -*- coding: utf-8 -*-

import numpy as np
import tqdm


class DefDict(dict):
    defaults = {}

    def __init__(self, **kwargs):
        i = {key: kwargs.get(key, val) for key, val in self.defaults.items()}
        super().__init__(**i)
        self.__dict__ = self

    def __str__(self):
        return '{name}({keys})'.format(name=self.__class__.__name__,
                                       keys=', '.join(self.defaults.keys()))

    __repr__ = __str__


def trange(n_iter, **kwargs):
    """
    Wrapper for ``tqdm.trange`` with some default options set

    Parameters
    ----------
    n_iter : int
        Number of iterations for progress bar

    Returns
    -------
    tqdm.tqdm instance
    """

    form = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
    return tqdm.trange(n_iter, ascii=True, leave=False,
                       bar_format=form, **kwargs)


def xcorr(X, Y, groups=None):
    """
    Calculates the cross-covariance matrix of ``X`` and ``Y``

    Parameters
    ----------
    X : (N x J) array_like
    Y : (N x K) array_like
    groups : (N,) array_like, optional
        Grouping array, where ``len(np.unique(groups))`` is the number of
        distinct groups in ``X`` and ``Y``. Cross-covariance matrices are
        computed separately for each group and are stacked row-wise.

    Returns
    -------
    (K[*G] x J) np.ndarray
        Cross-covariance of ``X`` and ``Y``
    """

    if groups is None:
        return _compute_xcorr(X, Y)
    else:
        return np.row_stack([_compute_xcorr(X[groups == grp],
                                            Y[groups == grp])
                             for grp in np.unique(groups)])


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


def dummy_code(groups, n_cond=1):
    """
    Dummy codes ``groups``

    Parameters
    ----------
    groups : (G,) list
        List with number of subjects in each of ``G`` groups
    n_cond : int
        Number of conditions, for each subject

    Returns
    -------
    Y : (N x G x C) np.ndarray
        Dummy coded group array
    """

    length = sum(groups) * n_cond
    width = len(groups) * n_cond
    Y = np.zeros((length, width))

    cstart = 0  # starting index for columns
    rstart = 0  # starting index for rows

    for i, grp in enumerate(groups):
        vals = np.repeat(np.eye(n_cond), grp).reshape(
            (n_cond, n_cond * grp)).T
        Y[:, cstart:cstart + n_cond][rstart:rstart + (grp * n_cond)] = vals

        cstart += vals.shape[1]
        rstart += vals.shape[0]

    return Y


def permute_cols(x, seed=None):
    """
    Permutes the rows for each column in ``x`` separately

    Taken directly from https://stackoverflow.com/a/27489131

    Parameters
    ----------
    x : (N x M) array_like
        Array to be permuted
    seed : {int, RandomState instance, None}, optional
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, ``seed`` is the seed used by the random number
        generator. If RandomState instance, ``seed`` is the random number
        generator. If None, the random number generator is the RandomState
        instance used by ``np.random``. Default: None

    Returns
    -------
    permuted : np.ndarray
        Permuted array
    """

    rs = get_seed(seed)
    ix_i = rs.random_sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]
