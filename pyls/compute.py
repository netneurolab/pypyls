#!/usr/bin/env python

from itertools import repeat
import multiprocessing as mp
import sys
import numpy as np
from scipy.stats import sem
from sklearn.utils.extmath import randomized_svd
from pyls.utils import xcorr, normalize


def mcsvd(X, Y):
    """
    Runs SVD on a mean-centered matrix computed from `X` and `Y`

    Parameters
    ----------
    X : (N x K) array_like
        Input array, where `N` is the number of subjects and `K` is the number
        of variables.
    Y : (N x J) array_like
        Dummy coded input array, where `N` is the number of subjects and `J`
        corresponds to the number of groups. A value of 1 in a given row/column
        indicates that a subject belongs to a given group.

    Returns
    -------
    U : (J x J-1) ndarray
        Left singular vectors
    d : (J-1 x J-1) ndarray
        Diagonal array of singular values
    V : (K x J-1) ndarray
        Right singular vectors
    """

    I = np.ones(shape=(len(Y), 1))
    M = np.linalg.inv(np.diag((I.T @ Y).flatten())) @ Y.T @ X
    L = np.ones(shape=(len(M), 1))
    R = M - L @ (((1/len(M)) * L.T) @ M)
    U, d, V = randomized_svd(R, n_components=Y.shape[-1]-1)

    return U, np.diag(d), V.T


def svd(X, Y, grouping=None):
    """
    Runs SVD on the cross-covariance matrix of `X` and `Y`

    Uses `randomized_svd` from `sklearn` for computation of a truncated SVD.
    Finds `L` components, where `L` is the minimum of the dimensions of `X` and
    `Y` if `X` and `Y` are 2-dimensionsal, or the minimum of the first and
    third dimensions of `X` and `Y` if `X` and `Y` are 3-dimensional

    If `X` and `Y` are 3-dimensional, the third dimension should represent a
    grouping factor. Cross-covariance matrices are computed separately for each
    group and then stacked prior to SVD.

    Parameters
    ----------
    X : (N x K) array_like
        Input array, where `N` is the number of subjects and `K` is the number
        of variables
    Y : (N x J) array_like
        Input array, where `N` is the number of subjects and `J` is the number
        of variables

    Returns
    -------
    U : (K x L) ndarray
        Left singular vectors
    d : (L x L) ndarray
        Diagonal array of singular values
    V : (J x L) ndarray
        Right singular vectors
    """

    if X.ndim != Y.ndim:
        raise ValueError("Number of dimensions of `X` and `Y` must match.")
    if X.ndim not in [2, 3]:
        raise ValueError("X must have 2 or 3 dimensions.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The first dimension of `X` and `Y` must match.")

    if X.ndim == 2:
        n_comp = min(min(X.shape), min(Y.shape))
    else:
        sl = slice(0, 3, 2)
        n_comp = min(min(X.shape[sl]), min(Y.shape[sl]))

    crosscov = normalize(xcorr(X, Y))

    U, d, V = randomized_svd(crosscov, n_components=n_comp)

    return U, np.diag(d), V.T


def split_half(X, Y, n_split=100, seed=None):
    """
    Parameters
    ----------
    X : (N x K [x G]) array_like
        Input array, where `N` is the number of subjects, `K` is the number of
        variables, and `G` is a grouping factor (if there are multiple groups)
    Y : (N x J [x G]) array_like
        Input array, where `N` is the number of subjects, `J` is the number of
        variables, and `G` is a grouping factor (if there are multiple groups)
    n_split : int, optional
        Number of split-half resamples during permutation testing. Default: 100
    seed : int, optional
        Whether to set random seed for reproducibility. Default: None

    Returns
    -------
    ucorr : (L,) ndarray
        Average correlation of left singular vectors across split-halves
    vcorr : (L,) ndarray
        Average correlation of right singular vectors across split-halves
    """

    if seed is not None: np.random.seed(seed)

    U, d, V = svd(X, Y)
    di = np.linalg.inv(d)
    vd, ud = V @ di, U @ di

    ucorr = np.zeros((n_split, U.shape[-1]))
    vcorr = np.zeros((n_split, V.shape[-1]))

    for n in range(n_split):
        split = np.zeros(len(X), dtype='bool')
        split[np.random.choice(len(X), size=len(X)//2, replace=False)] = True

        D1, D2 = xcorr(X[split], Y[split]), xcorr(X[~split], Y[~split])

        U1, U2 = D1 @ vd, D2 @ vd
        V1, V2 = D1.T @ ud, D2.T @ ud

        ucorr[n] = [np.corrcoef(u1, u2)[0, 1] for (u1, u2) in zip(U1.T, U2.T)]
        vcorr[n] = [np.corrcoef(v1, v2)[0, 1] for (v1, v2) in zip(V1.T, V2.T)]

    return ucorr.mean(axis=0), vcorr.mean(axis=0)


def procrustes(original, permuted, singular):
    """
    Performs Procrustes rotation on `permuted` to align with `original`

    `original` and `permuted` should be either left *or* right singular vectors
    from two SVDs. `singular` should be the diagonal matrix of singular values
    from the SVD that generated `permuted`.

    Parameters
    ----------
    original : array_like
    permuted : array_like
    singular : array_like

    Returns
    -------
    ndarray
        Singular values of rotated `permuted` matrix
    """

    N, _, P = np.linalg.svd(original.T @ permuted)
    Q = N @ P
    resamp = permuted @ singular @ Q

    return resamp, Q


def parallel_permute(X, Y,
                     original,
                     n_perm=1000,
                     n_split=None,
                     n_proc=None):
    """
    Parallelizes `single_perm()`` to `n_procs`

    Uses `starmap_async` with `multiprocessing.Pool()` to parallelize jobs.
    Each job will get a unique random seed to avoid re-use.

    Parameters
    ----------
    X : (N x K [x G]) array_like
    Y : (N x J [x G]) array_like
    original : array_like
        `U` or `V` from original SVD for use in procrustes rotation
    n_perm : int, optional
        Number of permutations to run. Default: 1000
    n_split : int, optional
        Number of split-half resamples to run. Default: None
    n_proc : int, optional
        Number of processes to use. Default: `mp.cpu_count()`

    Returns
    -------
    ndarray
        Distributions of singular values
    """

    def callback(result):
        permuted_values.append(result)

    if n_proc is None or n_proc < 0: n_proc = mp.cpu_count()

    permuted_values = []

    pool = mp.Pool(n_proc)
    pool.starmap_async(single_perm,
                       zip(repeat(X), repeat(Y), repeat(original),
                           repeat(n_split), np.arange(n_perm)),
                       callback=callback)
    pool.close()
    pool.join()

    permuted_values = np.asarray(permuted_values)

    if n_split is not None:
        permuted_values = permuted_values.transpose(0, 2, 1)

    return permuted_values


def serial_permute(X, Y,
                   original,
                   n_perm=1000,
                   n_split=None,
                   verbose=False):
    """
    Computes `perms` instances of `single_perm()`` in serial

    Parameters
    ----------
    X : (N x K [x G]) array_like
    Y : (N x J [x G]) array_like
    original : array_like
        `U` or `V` from original SVD for use in procrustes rotation
    n_perm : int, optional
        Number of permutations to run. Default: 1000
    n_split : int, optional
        Number of split-half resamples to run. Default: None
    verbose : bool, optional
        Whether to print status updates. Default: False

    Returns
    -------
    ndarray
        Distributions of singular values
    """

    permuted_values = []
    msg = ''

    for n in range(n_perm):
        if verbose:
            sys.stdout.write('\b'*len(msg))
            msg = 'Running permutation: {:>4}'.format(n+1)
            sys.stdout.write(msg)
            sys.stdout.flush()

        permuted_values.append(single_perm(X, Y, original, n_split=n_split))

    if verbose: sys.stdout.write('\n')

    permuted_values = np.asarray(permuted_values)

    if n_split is not None:
        permuted_values = permuted_values.transpose(0, 2, 1)

    return permuted_values


def single_perm(X, Y, original, n_split=None, seed=None):
    """
    Permutes `X` (w/o replacement) and recomputes SVD of `Y.T` @ `X`

    Uses procrustes rotation to ensure SVD is in same space as `original`

    Parameters
    ----------
    X : (N x K [x G]) array_like
    Y : (N x J [x G]) array_like
    original : array_like
        `U or `V` from original SVD for use in procrustes rotation
    n_split : int, optional
        Number of split-half resamples to run. Default: None
    seed : int, optional
        Whether to set random seed for reproducibility. Default: None

    Returns
    -------
    ndarray
        Distributions of singular values
    """

    if seed is not None: np.random.seed(seed)

    if X.ndim == 2:
        X_perm = np.random.permutation(X)
    else:
        while True:
            X_perm = perm_3d(X)
            if not np.allclose(X_perm.mean(axis=0), X.mean(axis=0)): break

    if n_split is not None:
        ucorr, vcorr = split_half(X_perm, Y, n_split=n_split, seed=seed)
        return ucorr, vcorr

    U, d, V = svd(X_perm, Y)

    if len(U) < len(V): permuted = U
    else: permuted = V

    resamp, *rest = procrustes(original, permuted, d)

    return np.sqrt((resamp**2).sum(axis=0))


def perm_3d(X):
    """
    Permutes `X` across 3rd dimension (i.e., groups)

    Parameters
    ----------
    X : (N x K [x G]) array_like

    Returns
    -------
    (N x K [x G]) ndarray
        Permuted `X`
    """

    X_2d = X.transpose((0, 2, 1)).reshape(X.shape[1], -1)
    X_2dp = np.random.permutation(X_2d)
    X_3dp = X_2dp.reshape(X.shape[-1], X.shape[0], -1)
    X_perm = X_3dp.transpose((1, 2, 0))

    return X_perm


def bootstrap(X, Y,
              U_orig, V_orig,
              n_boot=500,
              verbose=False):
    """
    Bootstrap `X`/`Y` (with replacement) and computes SE of singular values

    Parameters
    ----------
    X : (N x K [x G]) array_like
    Y : (N x J [x G]) array_like
    U_orig : (K[*G] x L) array_like
        Right singular vectors from original SVD
    V_orig : (J x L) array_like
        Left singular vectors from original SVD
    n_boot : int, optional
        Number of boostraps to run. Default: 500
    verbose : bool, optional
        Whether to print status updates. Default: False

    Returns
    -------
    (K[*G] x L x N_BOOT) ndarray
        Left singular vectors
    (J x L x N_BOOT) ndarray
        Right singular vectors
    """

    U_boot = np.zeros(U_orig.shape + (n_boot,))
    V_boot = np.zeros(V_orig.shape + (n_boot,))
    I = np.identity(U_orig.shape[1])
    msg = ''

    for i in range(n_boot):
        if verbose:
            sys.stdout.write('\b'*len(msg))
            msg = 'Running bootstrap: {:>6}'.format(i+1)
            sys.stdout.write(msg)
            sys.stdout.flush()

        inds = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
        X_boot, Y_boot = X[inds], Y[inds]
        U, d, V = svd(X_boot, Y_boot)

        U_boot[:, :, i], Q = procrustes(U_orig, U, I)
        V_boot[:, :, i] = V @ Q

    if verbose: sys.stdout.write('\n')

    return U_boot, V_boot


def perm_sig(permuted_singular, orig_singular):
    """
    Calculates significance of `orig_singular` values

    Compares amplitude of each singular value to distribution created via
    permutation in `permuted_singular`

    Parameters
    ----------
    permuted_singular : (NP x L) array_like
        Distribution of singular values from permutation testing where `NP` is
        the number of permutations and `L` is the number of components from the
        SVD
    orig_singular : (L x L) array_like
        Diagonal matrix of singular values from original SVD

    Returns
    -------
    array
        P-values of singular values from original SVD
    """

    pvals = np.zeros(len(orig_singular))
    n_perm = len(permuted_singular)

    for i in range(len(pvals)):
        top_dist = np.argwhere(permuted_singular[:, i] > orig_singular[i, i])
        pvals[i] = top_dist.size / n_perm

    return pvals


def boot_ci(U_boot, V_boot, p=0.05):
    """
    Generates CI for bootstrapped values `U_boot` and `V_boot`

    Parameters
    ----------
    U_boot : (K[*G] x L x N_BOOT) array_like
    V_boot : (J x L x N_BOOT) array_like
    p : float (0,1), optional
        Determines bounds of CI. Default: 0.05

    Returns
    -------
    (K[*G] x L x 2) ndarray
        CI for `U`
    (J x L x 2) array
        CI for `V`
    """

    low = 100 * (p / 2)
    high = 100 - low

    U_out = np.zeros(U_boot.shape[0:2] + (2,))
    V_out = np.zeros(V_boot.shape[0:2] + (2,))

    for n, f in enumerate([low, high]):
        U_out[:, :, n] = np.percentile(U_boot, f, axis=2)
        V_out[:, :, n] = np.percentile(V_boot, f, axis=2)

    return U_out, V_out


def boot_rel(U_orig, V_orig, U_boot, V_boot):
    """
    Determines bootstrap ratios (BSR) of saliences from bootstrap distributions

    Parameters
    ----------
    U_orig : (K[*G] x L) array_like
    V_orig : (J x L) array_like
    U_boot : (K[*G] x L x N_BOOT) array_like
    V_boot : (J x L x N_BOOT) array_like

    Returns
    -------
    (K[*G] x L) ndarray
        BSR for `U`
    (J x L) ndarray
        BSR for `V`
    """

    U_rel = U_orig / sem(U_boot, axis=-1)
    V_rel = V_orig / sem(V_boot, axis=-1)

    return U_rel, V_rel


def crossblock_cov(singular):
    """
    Calculates cross-block covariance of `singular` values

    Parameters
    ----------
    singular : (L x L) array_like
        Diagonal matrix of singular values from original SVD

    Returns
    -------
    (L,) ndarray
        Cross-block covariance
    """

    squared_sing = np.diag(singular)**2

    return squared_sing / squared_sing.sum()


def kaiser_criterion(singular):
    """
    Determines if variance explained by `singular` value > Kaiser criterion

    Kaiser criterion is 1/# singular values. If cross-block covariance
    explained by singular value exceeds criterion, return True; else, return
    False.

    Parameters
    ----------
    singular : (L x L) array_like
        Diagonal matrix of singular values from original SVD

    Returns
    -------
    (L,) ndarray
        Boolean array
    """

    return crossblock_cov(singular) > (1 / len(singular))


def boot_sig(boot):
    """
    Determines which entries of `boot` are significant via CI crossing

    If CI crosses zero, then bootstrap value is not

    Parameters
    ----------
    boot : (K[*G] x L x 2) array_like
        Components x confidence interval

    Returns
    -------
    (K,) ndarray
        Boolean array
    """

    return np.sign(boot).sum(axis=-1).astype('bool')
