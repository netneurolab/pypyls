#!/usr/bin/env python

import multiprocessing as mp
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.stats import sem
from pyls.utils import xcorr


def svd(X, Y, n_comp=None):
    """
    Runs SVD on the cross-covariance matrix of `X` and `Y`

    Uses `sklearn.utils.extmath.randomized_svd` for computation of a truncated
    SVD. Only returns first `n_comp` singular vectors/values.

    If `n_comp` is omitted, it is calculated as the minimum of the 1st and 2nd
    dimensions of `X` and `Y` (if `X`/`Y` are 2D) or the minimum of the 1st and
    3rd dimensions (if `X`/`Y` are 3D).

    Parameters
    ----------
    X : (N x K [x G]) array_like
        Input array, where N is the number of subjects, K is the number of
        variables, and G is a grouping factor (if there are multiple groups)
    Y : (N x J [x G]) array_like
        Input array, where N is the number of subjects, K is the number of
        variables, and G is a grouping factor (if there are multiple groups)
    n_comp : int, optional
        Rank of cross-covariance matrix; determines # of singular vectors.
        Default: rank of `Y.T @ X`

    Returns
    -------
    (K[*G] x N_COMP] ndarray
        Left singular vectors
    (N_COMP x N_COMP) ndarray
        Diagonal array of singular value
    (J[*G] x N_COMP) ndarray
        Right singular vectors
    """

    if X.ndim != Y.ndim:
        raise ValueError("Dimensions of `X` and `Y` must match.")
    if X.ndim not in [2,3]:
        raise ValueError("X must have 2 or 3 dimensions.")

    if X.ndim == 3: sl = slice(0,3,2)

    if n_comp is None:
        if X.ndim == 2: n_comp = min(min(X.shape), min(Y.shape))
        else: n_comp = min(min(X.shape[sl]), min(Y.shape[sl]))

    if X.ndim == 2 and n_comp > min(min(X.shape), min(Y.shape)):
        raise ValueError("Supplied `n_comp` is > rank of supplied matrices.")
    elif X.ndim == 3 and n_comp > min(min(X.shape[sl]), min(Y.shape[sl])):
        raise ValueError("Supplied `n_comp` is > rank of supplied matrices.")

    if X.ndim == 2:
        covarr = xcorr(X, Y)
    else:
        crosscov = []
        for group in range(X.shape[-1]):
            crosscov.append(xcorr(X[:,:,group], Y[:,:,group]))
        covarr = np.row_stack(crosscov)

    U, d, V = randomized_svd(covarr, n_components=n_comp)

    return U, np.diag(d), V.T


def procrustes(original, permuted, singular):
    """
    Performs Procrustes rotation on `permuted` to align with `original`

    `original` and `permuted` should be either left *or* right singular vectors
    from two SVDs. `singular` should be the diagonal matrix of singular values
    from the SVD that generated `original`.

    Parameters
    ----------
    original : array_like
    permuted : array_like
    singular : array_like

    Returns
    -------
    array
        Singular values of rotated `permuted` matrix
    """

    N, _, P = np.linalg.svd(original.T @ permuted)
    Q = N @ P
    resamp = permuted @ singular @ Q

    return resamp, Q


def parallel_permute(X, Y, n_comp, original, n_perm=1000, n_proc=1):
    """
    Parallelizes `single_perm()`` to `n_procs`

    Uses `apply_async` with `multiprocessing.Pool()``. This is really only
    useful if the SVD call from the permutation takes longer than the spawning
    of new processes -- which depends on how big your arrays are!

    Parameters
    ----------
    X : array_like
        Array of size (N x k)
    Y : array_like
        Array of size (N x j)
    n_comp : int
        Rank of cross-covariance matrix; determines # of singular vectors
    original : array_like
        `U` or `V` from original SVD for use in procrustes rotation
    n_perm : int, optional
        Number of permutations to run. Default: 1000
    n_proc : int, optional
        Number of processors to utilize. Default: 1

    Returns
    -------
    array
        Distributions of singular values
    """

    def callback(result):
        permuted_values.append(result)

    permuted_values = []
    pool = mp.Pool(n_proc)
    for i in range(n_perm):
        pool.apply_async(single_perm,
                         args=(X,Y,n_comp,original),
                         kwds={'seed':np.random.randint(4294967295)},
                         callback=callback)
    pool.close()
    pool.join()

    return np.array(permuted_values)


def serial_permute(X, Y, n_comp, original, n_perm=1000, n_split=None):
    """
    Computes `perms` instances of `single_perm()`` in serial

    Parameters
    ----------
    X : array_like
        Array of size (N x k)
    Y : array_like
        Array of size (N x j)
    n_comp : int
        Rank of cross-covariance matrix; determines # of singular vectors
    original : array_like
        `U` or `V` from original SVD for use in procrustes rotation
    n_perm : int, optional
        Number of permutations to run. Default: 1000
    n_split : int, optional
        Number of split-half resamples to assess reliability. Default: None

    Returns
    -------
    array
        Distributions of singular values
    """

    permuted_values = np.zeros((n_perm,n_comp))

    for n in range(n_perm):
        permuted_values[n] = single_perm(X, Y, n_comp, original,
                                         n_split=n_split)

    return permuted_values


def single_perm(X, Y, n_comp, original, n_split=None, seed=None):
    """
    Permutes `X` (w/o replacement) and recomputes SVD of `Y.T` @ `X`

    Uses procrustes rotation to ensure SVD is in same space as `original`

    Parameters
    ----------
    X : array_like
        Array of size (N x k)
    Y : array_like
        Array of size (N x j)
    n_comp : int
        Rank of cross-covariance matrix; determines # of singular vectors
    original : array_like
        `U or `V` from original SVD for use in procrustes rotation
    n_split : int, optional
        Number of split-half resamples to assess reliability. Default: None
    seed : int, optional
        To set `np.random.seed`. Default: None

    Returns
    -------
    array
        Distributions of singular values
    """

    if seed is not None: np.random.seed(seed)

    while True:
        if X.ndim == 2: X_perm = np.random.permutation(X)
        else: X_perm = perm_3d(X)

        if not np.allclose(X_perm.mean(axis=0), X.mean(axis=0)): break

    if n_split is not None:
        pass
    else:
        U, d, V = svd(X_perm, Y, n_comp)

        if len(U) < len(V): permuted = U
        else: permuted = V

        resamp, *rest = procrustes(original, permuted, d)

        return np.sqrt((resamp**2).sum(axis=0))


def perm_3d(X):
    """
    Permutes `X` across 3rd dimension (i.e., groups)

    Parameters
    ----------
    X : array_like
        Array of size (N x k x group)

    Returns
    -------
    array
        Permuted `X`
    """

    X_2d   = X.transpose((0,2,1)).reshape(X.shape[1],-1)
    X_2dp  = np.random.permutation(X_2d)
    X_3dp  = X_2dp.reshape(X.shape[-1],X.shape[0],-1)
    X_perm = X_3dp.transpose((1,2,0))

    return X_perm


def bootstrap(X, Y, n_comp, U_orig, V_orig, n_boot=500):
    """
    Bootstrap `X`/`Y` (with replacement) and computes SE of singular values

    Parameters
    ----------
    X : array_like
        Array of size (N x k)
    Y : array_like
        Array of size (N x j)
    n_comp : int
        Rank of cross-covariance matrix; determines # of singular vectors
    U_orig : array_like
        Array of size (k x n_comp)
    V_orig : array_like
        Array of size (j x n_comp)
    n_boot : int, optional
        Number of boostraps to run. Default: 500

    Returns
    -------
    array
        Left singular vectors (k x n_comp x n_boot)
    array
        Right singular vectors (j x n_comp x n_boot)
    """

    U_boot = np.zeros(U_orig.shape + (n_boot,))
    V_boot = np.zeros(V_orig.shape + (n_boot,))
    I = np.identity(n_comp)

    for i in range(n_boot):
        inds = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
        X_boot, Y_boot = X[inds], Y[inds]
        U, d, V = svd(X_boot, Y_boot, n_comp)

        U_boot[:,:,i], Q = procrustes(U_orig, U, I)
        V_boot[:,:,i] = V @ Q

    return U_boot, V_boot


def perm_sig(permuted_singular, orig_singular):
    """
    Calculates significance of `orig_singular` vaues

    Compares amplitude of each singular value to distribution created via
    permutation in `permuted_singular`

    Parameters
    ----------
    permuted_singular : array_like
        Distribution of singular values (n_perms x n_comp) from permutation
        testing
    orig_singular : array_like
        Singular values (diagonal, n_comp x n_comp) from original SVD

    Returns
    -------
    array
        P-values of singular values from original SVD
    """

    pvals = np.zeros(len(orig_singular))
    n_perm = len(permuted_singular)

    for i in range(len(pvals)):
        top_of_dist = np.argwhere(permuted_singular[:,i] > orig_singular[i,i])
        pvals[i] = top_of_dist.size/n_perm

    return pvals


def boot_ci(U_boot, V_boot, p=0.05):
    """
    Generates CI for bootstrapped values `U_boot` and `V_boot`

    Parameters
    ----------
    U_boot : array_like
        Array of size (k x n_comp x n_boot)
    V_boot : array_like
        Array of size (j x n_comp x n_boot)
    p : float (0,1), optional
        Determines bounds of CI. Default: 0.05

    Returns
    -------
    array
        CI for U (k x n_comp x 2)
    array
        CI for V (j x n_comp x 2)
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
    Determines bootstrap ratios (BSR) of saliences from bootstrap distributions

    Parameters
    ----------
    U_orig : array_like
        Array of size (k x n_comp)
    V_orig : array_like
        Array of size (j x n_comp)
    U_boot : array_like
        Array of size (k x n_comp x n_boot)
    V_boot : array_like
        Array of size (j x n_comp x n_boot)

    Returns
    -------
    array
        BSR for `U` (k x n_comp)
    array
        BSR for `V` (j x n_comp)
    """

    U_rel = U_orig/sem(U_boot,axis=-1)
    V_rel = V_orig/sem(V_boot,axis=-1)

    return U_rel, V_rel


def crossblock_cov(singular):
    """
    Calculates cross-block covariance of `singular` values

    Parameters
    ----------
    singular : array_like
        Singular values (diagonal, n_comp x n_comp) from SVD

    Returns
    -------
    array
        Cross-block covariance (n_comp,)
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
    singular : array_like
        Singular values (diagonal, n_comp x n_comp) from SVD

    Returns
    -------
    array
        Boolean array (n_comp,)
    """

    return crossblock_cov(singular) > (1/len(singular))


def boot_sig(boot):
    """
    Determines which entries of `boot` are significant via CI crossing

    If CI crosses zero, then bootstrap value is not

    Parameters
    ----------
    boot : array_like
        Components x confidence interval (k x 2)

    Returns
    -------
    array
        Boolean array (k,)
    """

    return np.sign(boot).sum(axis=-1)
