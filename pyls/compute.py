# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_array
from pyls import utils


def xcorr(X, Y, norm=True, covariance=False):
    """
    Calculates the cross-covariance matrix of `X` and `Y`

    Parameters
    ----------
    X : (S, B) array_like
        Input matrix, where `S` is samples and `B` is features.
    Y : (S, T) array_like, optional
        Input matrix, where `S` is samples and `T` is features.
    norm : bool, optional
        Whether to normalize `X` and `Y` (i.e., sum of squares = 1). Default:
        True
    covariance : bool, optional
        Whether to calculate the cross-covariance matrix instead of the cross-
        correlation matrix. Default: False

    Returns
    -------
    xprod : (T, B) `numpy.ndarray`
        Cross-covariance of `X` and `Y`
    """

    if X.ndim != Y.ndim:
        raise ValueError('Number of dims of `X` and `Y` must match.')
    if X.ndim != 2:
        raise ValueError('`X` and `Y` must each have 2 dims.')
    if len(X) != len(Y):
        raise ValueError('The first dim of `X` and `Y` must match.')

    if not covariance:
        Xn, Yn = zscore(X), zscore(Y)
    else:
        Xn, Yn = X - X.mean(0, keepdims=True), Y - Y.mean(0, keepdims=True)

    if norm:
        Xn, Yn = normalize(Xn), normalize(Yn)
    xprod = (Yn.T @ Xn) / (len(Xn) - 1)

    return xprod


def zscore(data, axis=0, ddof=1, comp=None):
    """
    Z-scores `X` by subtracting mean and dividing by standard deviation

    Effectively the same as `np.nan_to_num(scipy.stats.zscore(X))` but
    handles DivideByZero without issuing annoying warnings.

    Parameters
    ----------
    data : (N, ...) array_like
        Data to be z-scored
    axis : int, optional
        Axis along which to z-score. Default: 0
    ddof : int, optional
        Delta degrees of freedom.  The divisor used in calculations is
        `M - ddof`, where `M` is the number of elements along `axis` in
        `comp`. Default: 1
    comp : (M, ...) array_like
        Distribution to z-score `data`. Should have same dimension as `data`
        along every axis except `axis`. If not provided, `data` will be used.
        Default: None

    Returns
    -------
    zarr : (N, ...) `numpy.ndarray`
        Z-scored version of `data`
    """

    data = check_array(data, ensure_2d=False, allow_nd=True)

    # check if z-score against another distribution or self
    if comp is not None:
        comp = check_array(comp, ensure_2d=False, allow_nd=True)
    else:
        comp = data

    avg = comp.mean(axis=axis, keepdims=True)
    stdev = comp.std(axis=axis, ddof=ddof, keepdims=True)

    # avoid DivideByZero errors
    zeros = stdev == 0
    if np.any(zeros):
        avg[zeros] = 0
        stdev[zeros] = 1

    zarr = (data - avg) / stdev
    zarr[np.repeat(zeros, zarr.shape[axis], axis=axis)] = 0

    return zarr


def normalize(X, axis=0):
    """
    Normalizes `X` along `axis`

    Utilizes Frobenius norm (or Hilbert-Schmidt norm / `L_{p,q}` norm where
    `p=q=2`)

    Parameters
    ----------
    X : (S, B) array_like
        Input array
    axis : int, optional
        Axis for normalization. Default: 0

    Returns
    -------
    normed : (S, B) `numpy.ndarray`
        Normalized `X`
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


def rescale_test(X_train, X_test, Y_train, U, V):
    """
    Generates out-of-sample predicted `Y` values

    Parameters
    ----------
    X_train : (S1, B) array_like
        Data matrix, where `S1` is observations and `B` is features
    X_test : (S2, B)
        Data matrix, where `S2` is observations and `B` is features
    Y_train : (S1, T) array_like
        Behavioral matrix, where `S1` is observations and `T` is features

    Returns
    -------
    Y_pred : (S2, T) `numpy.ndarray`
        Behavioral matrix, where `S2` is observations and `T` is features
    """

    X_resc = zscore(X_test, comp=X_train, axis=0, ddof=1)
    Y_pred = (X_resc @ U @ V.T) + Y_train.mean(axis=0, keepdims=True)

    return Y_pred


def perm_sig(orig, perm):
    """
    Calculates significance of `orig` values agains `perm` distributions

    Compares amplitude of each singular value to distribution created via
    permutation in `perm`

    Parameters
    ----------
    orig : (L, L) array_like
        Diagonal matrix of singular values for `L` latent variables
    perm : (L, P) array_like
        Distribution of singular values from permutation testing where `P` is
        the number of permutations

    Returns
    -------
    sprob : (L,) `numpy.ndarray`
        Number of permutations where singular values exceeded original data
        decomposition for each of `L` latent variables normalized by the total
        number of permutations. Can be interpreted as the statistical
        significance of the latent variables (i.e., non-parametric p-value).
    """

    sp = np.sum(perm > np.diag(orig)[:, None], axis=1) + 1
    sprob = sp / (perm.shape[-1] + 1)

    return sprob


def boot_ci(boot, ci=95):
    """
    Generates CI for bootstrapped values `boot`

    Parameters
    ----------
    boot : (G, L, B) array_like
        Singular vectors, where `G` is features, `L` is components, and `B` is
        bootstraps
    ci : (0, 100) float, optional
        Confidence interval bounds to be calculated. Default: 95

    Returns
    -------
    lower : (G, L) `numpy.ndarray`
        Lower bound of CI for singular vectors in `boot`
    upper : (G, L) `numpy.ndarray`
        Upper bound of CI for singular vectors in `boot`
    """

    low = (100 - ci) / 2
    prc = [low, 100 - low]

    lower, upper = np.percentile(boot, prc, axis=-1)

    return lower, upper


def boot_rel(orig, u_sum, u_square, n_boot):
    """
    Determines bootstrap ratios (BSR) of saliences from bootstrap distributions

    Parameters
    ----------
    orig : (G, L) array_like
        Original singular vectors
    u_sum : (G, L) array_like
        Sum of bootstrapped singular vectors
    u_square : (G, L) array_like
        Sum of squared bootstraped singular vectors
    n_boot : int
        Number of bootstraps used in generating `u_sum` and `u_square`

    Returns
    -------
    bsr : (G, L) `numpy.ndarray`
        Bootstrap ratios for provided singular vectors
    """

    u_sum2 = (u_sum ** 2) / n_boot
    u_se = np.sqrt(np.abs(u_square - u_sum2) / (n_boot - 1))
    bsr = orig / u_se

    return bsr, u_se


def procrustes(original, permuted, singular):
    """
    Performs Procrustes rotation on `permuted` to align with `original`

    `original` and `permuted` should be either left *or* right singular
    vector from two SVDs. `singular` should be the diagonal matrix of
    singular values from the SVD that generated `original`

    Parameters
    ----------
    original : array_like
    permuted : array_like
    singular : array_like

    Returns
    -------
    resamp : `numpy.ndarray`
        Singular values of rotated `permuted` matrix
    rotate : `numpy.ndarray`
        Matrix for rotating `permuted` to `original`
    """

    temp = original.T @ permuted
    N, _, P = randomized_svd(temp, n_components=min(temp.shape))
    rotate = P.T @ N.T
    resamp = permuted @ singular @ rotate

    return resamp, rotate


def get_group_mean(X, Y, n_cond=1, mean_centering=0):
    """
    Parameters
    ----------
    X : (S, B) array_like
        Input data matrix, where `S` is observations and `B` is features
    Y : (S, T) array_like, optional
        Dummy coded input array, where `S` is observations and `T`
        corresponds to the number of different groups x conditions. A value
        of 1 indicates that an observation belongs to a specific group or
        condition.
    n_cond : int, optional
        Number of conditions in dummy coded `Y` array. Default: 1
    mean_centering : {0, 1, 2}, optional
        Mean-centering method. Default: 0

    Returns
    -------
    group_mean : (T, B) `numpy.ndarray`
        Means to be removed from `X` during centering
    """

    if mean_centering == 0:
        # we want means of GROUPS, collapsing across conditions
        inds = slice(0, Y.shape[-1], n_cond)
        groups = utils.dummy_code(Y[:, inds].sum(axis=0).astype(int) * n_cond)
    elif mean_centering == 1:
        # we want means of CONDITIONS, collapsing across groups
        groups = Y.copy()
    elif mean_centering == 2:
        # we want the overall mean of the entire dataset
        groups = np.ones((len(X), 1))
    else:
        raise ValueError("Mean centering type must be in [0, 1, 2].")

    # get mean of data over grouping variable
    group_mean = np.row_stack([X[grp].mean(axis=0)[None] for grp in
                               groups.T.astype(bool)])

    # we want group_mean to have the same number of rows as Y does columns
    # that way, we can easily subtract it for mean centering the data
    # and generating the matrix for SVD
    if mean_centering == 0:
        group_mean = np.repeat(group_mean, n_cond, axis=0)
    elif mean_centering == 1:
        group_mean = group_mean.reshape(-1, n_cond, X.shape[-1]).mean(axis=0)
        group_mean = np.tile(group_mean.T, int(Y.shape[-1] / n_cond)).T
    else:
        group_mean = np.repeat(group_mean, Y.shape[-1], axis=0)

    return group_mean


def get_mean_center(X, Y, n_cond=1, mean_centering=0, means=True):
    """
    Parameters
    ----------
    X : (S, B) array_like
        Input data matrix, where `S` is observations and `B` is features
    Y : (S, T) array_like, optional
        Dummy coded input array, where `S` is observations and `T`
        corresponds to the number of different groups x conditions. A value
        of 1 indicates that an observation belongs to a specific group or
        condition.
    n_cond : int, optional
        Number of conditions in dummy coded `Y` array. Default: 1
    mean_centering : {0, 1, 2}, optional
        Mean-centering method. Default: 0
    means : bool, optional
        Whether to return demeaned averages instead of demeaned data. Default:
        True

    Returns
    -------
    mean_centered : {(T, B), (S, B)} `numpy.ndarray`
        If `means` is True, returns array with shape (T, B); otherwise, returns
        (S, B)
    """

    mc = get_group_mean(X, Y, n_cond=n_cond, mean_centering=mean_centering)

    if means:
        # take mean of groups and subtract relevant mean_centering entry
        mean_centered = np.row_stack([X[grp].mean(axis=0) - mc[n] for (n, grp)
                                      in enumerate(Y.T.astype(bool))])
    else:
        # subtract relevant mean_centering entry from each observation
        mean_centered = np.row_stack([X[grp] - mc[n][None] for (n, grp)
                                      in enumerate(Y.T.astype(bool))])

    return mean_centered


def efficient_corr(x, y):
    """
    Computes correlation of matching columns in `x` and `y`

    Parameters
    ----------
    x, y : (N, M) array_like
        Input data arrays

    Returns
    -------
    corr : (M,) numpy.ndarray
        Correlations of columns in `x` and `y`
    """

    # we need 2D arrays
    x, y = np.vstack(x), np.vstack(y)

    # check shapes
    if x.shape != y.shape:
        if x.shape[-1] != 1 and y.shape[-1] != 1:
            raise ValueError('Provided inputs x and y must either have '
                             'matching shapes or one must be a column '
                             'vector.\nProvided data:\n\tx: {}\n\ty: {}'
                             .format(x.shape, y.shape))

    corr = np.sum(zscore(x, ddof=1) * zscore(y, ddof=1), axis=0) / (len(x) - 1)

    # fix rounding errors
    corr[corr > 1] = 1
    corr[corr < -1] = -1

    return corr
