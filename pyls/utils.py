# -*- coding: utf-8 -*-

import numpy as np
import tqdm
from sklearn.utils.validation import check_array, check_random_state


class DefDict(dict):
    """
    Subclass of dictionary that instantiates with ``self.defaults``
    """

    defaults = {}

    def __init__(self, **kwargs):
        i = {key: kwargs.get(key, val) for key, val in self.defaults.items()}
        super().__init__(**i)
        self.__dict__ = self

    def __str__(self):
        items = [k for k in self.keys() if self.get(k) is not None]
        return '{name}({keys})'.format(name=self.__class__.__name__,
                                       keys=', '.join(items))

    __repr__ = __str__


def check_xcorr_inputs(X, Y):
    """
    Ensures that ``X`` and ``Y`` are appropriate for use in ``xcorr()``

    Parameters
    ----------
    X : (S x B) array_like
        Input matrix, where ``S`` is samples and ``B`` is features.
    Y : (S x T) array_like, optional
        Input matrix, where ``S`` is samples and ``T`` is features.

    Raises
    ------
    ValueError
    """

    if X.ndim != Y.ndim:
        raise ValueError('Number of dims of ``X`` and ``Y`` must match.')
    if X.ndim != 2:
        raise ValueError('``X`` and ``Y`` must each have 2 dims.')
    if len(X) != len(Y):
        raise ValueError('The first dim of ``X`` and ``Y`` must match.')


def trange(n_iter, **kwargs):
    """
    Wrapper for ``tqdm.trange`` with some default options set

    Parameters
    ----------
    n_iter : int
        Number of iterations for progress bar

    Returns
    -------
    progbar : tqdm.tqdm instance
    """

    form = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
    return tqdm.trange(n_iter, ascii=True, leave=False,
                       bar_format=form, **kwargs)


def xcorr(X, Y, norm=True):
    """
    Calculates the cross-covariance matrix of ``X`` and ``Y``

    Parameters
    ----------
    X : (S x B) array_like
        Input matrix, where ``S`` is samples and ``B`` is features.
    Y : (S x T) array_like, optional
        Input matrix, where ``S`` is samples and ``T`` is features.

    Returns
    -------
    xprod : (T x B) np.ndarray
        Cross-covariance of ``X`` and ``Y``
    """

    check_xcorr_inputs(X, Y)
    Xn, Yn = zscore(X), zscore(Y)
    if norm:
        Xn, Yn = normalize(Xn), normalize(Yn)
    xprod = (Yn.T @ Xn) / (len(Xn) - 1)

    return xprod


def zscore(data, axis=0, ddof=1, comp=None):
    """
    Z-scores ``X`` by subtracting mean and dividing by standard deviation

    Effectively the same as ``np.nan_to_num(scipy.stats.zscore(X))`` but
    handles DivideByZero without issuing annoying warnings.

    Parameters
    ----------
    data : (N x ...) array_like
        Data to be z-scored
    axis : int, optional
        Axis to use to z-score data. Default: 0
    ddof : int, optional
        Delta degrees of freedom.  The divisor used in calculations is
        ``M - ddof``, where ``M`` is the number of elements along ``axis``
        in ``comp``. Default: 1
    comp : (M x ...) array_like
        Distribution to z-score ``data``. Should have same dimension as data
        along `axis`. Default: ``data``

    Returns
    -------
    zarr : (N x ...) np.ndarray
        Z-scored version of ``data``
    """

    data = check_array(data, ensure_2d=False, allow_nd=True)

    if comp is not None:
        comp = check_array(comp, ensure_2d=False, allow_nd=True)
    else:
        comp = data

    avg = comp.mean(axis=axis, keepdims=True)
    stdev = comp.std(axis=axis, ddof=ddof, keepdims=True)
    zeros = stdev == 0

    if np.any(zeros):
        avg[zeros] = 0
        stdev[zeros] = 1

    zarr = (data - avg) / stdev
    zarr[np.repeat(zeros, zarr.shape[axis], axis=axis)] = 0

    return zarr


def normalize(X, axis=0):
    """
    Normalizes ``X`` along ``axis``

    Utilizes Frobenius norm (or Hilbert-Schmidt norm, L_{p,q} norm where p=q=2)

    Parameters
    ----------
    X : (S x B) array_like
        Input array
    axis : int, optional
        Axis for normalization. Default: 0

    Returns
    -------
    normed : (S x B) np.ndarray
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


def dummy_code(groups, n_cond=1):
    """
    Dummy codes ``groups`` and ``n_cond``

    Parameters
    ----------
    groups : (G,) list
        List with number of subjects in each of ``G`` groups
    n_cond : int, optional
        Number of conditions, for each subject. Default: 1

    Returns
    -------
    Y : (S x F) np.ndarray
        Dummy-coded group array
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

    Taken from https://stackoverflow.com/a/27489131

    Parameters
    ----------
    x : (S x B) array_like
        Input array to be permuted
    seed : {int, RandomState instance, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    permuted : np.ndarray
        Permuted array
    """

    rs = check_random_state(seed)
    ix_i = rs.random_sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]
