"""Core computational functions for PLS analysis."""

import numpy as np
from scipy.stats import zscore, zmap
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_X_y, check_random_state
from pyls import utils


def svd(crosscov, n_components=None, seed=None):
    """
    Calculate the SVD of `crosscov` and returns singular vectors/values.

    Parameters
    ----------
    crosscov : (B, T) array_like
        Cross-covariance (or cross-correlation) matrix to be decomposed
    n_components : int, optional
        Number of components to retain from decomposition
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    U : (B, L) `numpy.ndarray`
        Left singular vectors from singular value decomposition
    d : (L, L) `numpy.ndarray`
        Diagonal array of singular values from singular value decomposition
    V : (J, L) `numpy.ndarray`
        Right singular vectors from singular value decomposition
    """
    seed = check_random_state(seed)
    crosscov = np.asanyarray(crosscov)

    if n_components is None:
        n_components = min(crosscov.shape)
    elif not isinstance(n_components, int):
        raise TypeError(
            "Provided `n_components` {} must be of type int".format(n_components)
        )

    # run most computationally efficient SVD
    if crosscov.shape[0] <= crosscov.shape[1]:
        U, d, V = randomized_svd(
            crosscov.T, n_components=n_components, random_state=seed, transpose=False
        )
        V = V.T
    else:
        V, d, U = randomized_svd(
            crosscov, n_components=n_components, random_state=seed, transpose=False
        )
        U = U.T

    return U, np.diag(d), V


def xcorr(X, Y, norm=False, covariance=False):
    """
    Calculate the cross-covariance matrix of `X` and `Y`.

    Parameters
    ----------
    X : (S, B) array_like
        Input matrix, where `S` is samples and `B` is features.
    Y : (S, T) array_like, optional
        Input matrix, where `S` is samples and `T` is features.
    norm : bool, optional
        Whether to normalize `X` and `Y` (i.e., sum of squares = 1). Default:
        False
    covariance : bool, optional
        Whether to calculate the cross-covariance matrix instead of the cross-
        correlation matrix. Default: False

    Returns
    -------
    xprod : (T, B) `numpy.ndarray`
        Cross-covariance of `X` and `Y`
    """
    check_X_y(X, Y, multi_output=True)

    # we could just use scipy.stats zscore but if we do this we retain the
    # original data structure; if pandas dataframes were given, a dataframe
    # will be returned
    if not covariance:
        Xn = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        Yn = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
    else:
        Xn, Yn = X - X.mean(0, keepdims=True), Y - Y.mean(0, keepdims=True)

    if norm:
        Xn, Yn = normalize(Xn), normalize(Yn)

    xprod = (Yn.T @ Xn) / (len(Xn) - 1)

    return xprod


def normalize(X, axis=0):
    """
    Normalize `X` along `axis`.

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
    Generate out-of-sample predicted `Y` values.

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
    X_resc = zmap(X_test, compare=X_train, ddof=1)
    Y_pred = (X_resc @ U @ V.T) + Y_train.mean(axis=0, keepdims=True)

    return Y_pred


def perm_sig(orig, perm):
    """
    Calculate significance of `orig` values agains `perm` distributions.

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
    Generate CI for bootstrapped values `boot`.

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
    Determine bootstrap ratios (BSR) of saliences from bootstrap distributions.

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
    u_sum2 = (u_sum**2) / n_boot
    u_se = np.sqrt(np.abs(u_square - u_sum2) / (n_boot - 1))
    bsr = orig / u_se

    return bsr, u_se


def procrustes(original, permuted, singular):
    """
    Perform Procrustes rotation on `permuted` to align with `original`.

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
    """
    temp = original.T @ permuted
    N, _, P = randomized_svd(temp, n_components=min(temp.shape))
    resamp = permuted @ singular @ (P.T @ N.T)

    return resamp


def get_group_mean(X, Y, n_cond=1, mean_centering=0):
    """
    Get means of `X` over groups defined in `Y` for mean-centering.

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
    group_mean = np.vstack([X[grp].mean(axis=0)[None] for grp in groups.T.astype(bool)])

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
    Get mean-centered data or means of `X` over groups defined in `Y`.

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
        mean_centered = np.vstack(
            [X[grp].mean(axis=0) - mc[n] for (n, grp) in enumerate(Y.T.astype(bool))]
        )
    else:
        # subtract relevant mean_centering entry from each observation
        mean_centered = np.vstack(
            [X[grp] - mc[n][None] for (n, grp) in enumerate(Y.T.astype(bool))]
        )

    return mean_centered


def efficient_corr(x, y):
    """
    Compute correlation of matching columns in `x` and `y`.

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
            raise ValueError(
                "Provided inputs x and y must either have "
                "matching shapes or one must be a column "
                "vector.\nProvided data:\n\tx: {}\n\ty: {}".format(x.shape, y.shape)
            )

    corr = np.sum(zscore(x, ddof=1) * zscore(y, ddof=1), axis=0) / (len(x) - 1)

    # fix rounding errors
    corr = np.clip(corr, -1, 1)

    return corr


def varexp(singular):
    """
    Calculate the variance explained by values in `singular`.

    Parameters
    ----------
    singular : (L, L) array_like
        Singular values from singular value decomposition

    Returns
    -------
    varexp : (L, L) `numpy.ndarray`
        Variance explained
    """
    if singular.ndim != 2:
        raise ValueError(
            "Provided `singular` array must be a square diagonal "
            "matrix, not array of shape {}".format(singular.shape)
        )

    return np.diag(np.diag(singular) ** 2 / np.sum(np.diag(singular) ** 2))
