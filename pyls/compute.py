#!/usr/bin/env python

import multiprocessing as mp
import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.stats import sem
from pyls.utils import xcorr


def svd(X, Y, k=None):
    """
    Runs SVD on the covariance matrix of `X` and `Y`

    Uses sklearn.utils.extmath.randomized_svd for computation of a truncated
    SVD. Only returns first `k` singular vectors/values.

    If `k` is omitted, it is calculated as the minimum of the dimensions of X
    and Y.

    Parameters
    ----------
    X, Y : array (N x j [x group]), array (N x k [x group])
    k : int
        rank of cross-covariance matrix; determines # of singular vectors

    Returns
    -------
    U, d, V : left singular vectors, singular values, right singular vectors
    """

    if k is None: k = min(min(X.shape),min(Y.shape))

    if k > min(min(X.shape),min(Y.shape)):
        raise ValueError("Supplied `k` is > rank of supplied matrices.")

    if X.ndim != Y.ndim:
        raise ValueError("Dimensions of `X` and `Y` must match.")
    if X.ndim not in [2,3]:
        raise ValueError("X must have 2 or 3 dimensions.")
    elif X.ndim == 3:
        crosscov = []
        for group in range(X.shape[-1]):
            crosscov.append(xcorr(X[:,:,group],Y[:,:,group]))
        covmat = np.row_stack(crosscov)
    else:
        covmat = xcorr(X, Y)

    U, d, V = randomized_svd(covmat, n_components=k)

    return U, np.diag(d), V.T


def procrustes(original, permuted, singular):
    """
    Performs Procrustes rotation on `permuted` to align with `original`

    Parameters
    ----------
    original : array
    permuted : array
    singular : array

    Returns
    -------
    array : singular values of rotated `permuted` matrix
    """

    N, _, P = np.linalg.svd(original.T @ permuted)
    Q = N @ P
    resamp = permuted @ singular @ Q

    return resamp, Q


def parallel_permute(X, Y, k, original, perms=1000, procs=1):
    """
    Parallelizes single_perm() to `procs`

    Uses apply_async with multiprocessing.Pool(). This is really only useful
    if the SVD call from the permutation would be #rough on memory

    Parameters
    ----------
    X : array (N x j)
    Y : array (N x t)
    k : int
        rank of Y.T @ X matrix
    original : array
        U or V from original SVD for use in procrustes rotation
    perms : int
        number of permutations to run
    procs : int
        function will multiprocess permutations for potential speed-up

    Returns
    -------
    array : distributions of singular values
    """

    def callback(result):
        permuted_values.append(result)

    permuted_values = []
    pool = mp.Pool(procs, maxtasksperchild=10)
    for i in range(perms):
        pool.apply_async(single_perm,
                         args=(X,Y,k,original),
                         kwds={'seed':np.random.randint(4294967295)},
                         callback=callback)
    pool.close()
    pool.join()

    return np.array(permuted_values)


def serial_permute(X, Y, k, original, perms=1000):
    """
    Computes `perms` instances of single_perm() in serial

    Parameters
    ----------
    X : array (N x j)
    Y : array (N x t)
    k : int
        rank of Y.T @ X matrix
    original : array
        U or V from original SVD for use in procrustes rotation
    perms : int
        number of permutations to run
    procs : int
        function will multiprocess permutations for potential speed-up

    Returns
    -------
    array : distributions of singular values
    """

    permuted_values = np.zeros((perms,k))

    for n in range(perms):
        permuted_values[n] = single_perm(X,Y,k,original)

    return permuted_values


def single_perm(data, behav, n_comp, orig, seed=None):
    """
    Permutes `data` (w/o replacement) and recomputes SVD of `behav`.T @ `data`

    Uses procrustes rotation to ensure SVD is in same space as `orig`

    Parameters
    ----------
    data : array (N x j)
    behav : array (N x t)
    n_comp : int
        rank of behav.T @ data matrix
    orig : array
        U or V from original SVD for use in procrustes rotation
    perms : int
        number of permutations to run

    Returns
    -------
    array : distributions of singular values
    """

    if seed is not None: np.random.seed(seed)
    X_perm = np.random.permutation(data)
    U, d, V = svd(X_perm, behav, n_comp)

    if len(U) < len(V): permuted = U
    else: permuted = V

    resamp, *rest = procrustes(orig, permuted, d)

    return np.sqrt((resamp**2).sum(axis=0))


def bootstrap(X, Y, k, U_orig, V_orig, boots=500, procs=1, verbose=False):
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
    verbose: bool
        whether to print occasional progress messages

    Returns
    -------
    array, array : left (t x k x boots), right (j x k x boots) singular vectors
    """

    U_boot = np.zeros(U_orig.shape + (boots,))
    V_boot = np.zeros(V_orig.shape + (boots,))
    I = np.identity(k)

    for i in range(boots):
        if verbose and i % 100 == 0: print("Bootstrap {}".format(str(i)))
        inds = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
        X_boot, Y_boot = X[inds], Y[inds]
        U, d, V = svd(X_boot, Y_boot, k)

        U_boot[:,:,i], Q = procrustes(U_orig, U, I)
        V_boot[:,:,i] = V @ Q

    return U_boot, V_boot


def perm_sig(permuted_svalues, orig_svalues):
    """
    Calculates significance of `orig_svalues`

    Compares amplitude of each singular value to distribution created via
    permutation in `permuted_svalues`

    Parameters
    ----------
    permuted_svalues : array (n_perms x k)
        distribution of singular values from permutation testing
    orig_svalues : array (diagonal, k x k)
        singular values from original SVD

    Returns
    -------
    array : p-values of singular values from original SVD
    """

    pvals = np.zeros(len(orig_svalues))
    n_perms = len(permuted_svalues)

    for i in range(len(pvals)):
        top_of_dist = np.where(permuted_svalues[:,i]>orig_svalues[i,i])[0]
        pvals[i] = top_of_dist.size/n_perms

    return pvals


def boot_ci(U_boot, V_boot, p=.01):
    """
    Generates CI for bootstrapped values `U_boot` and `V_boot`

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


def crossblock_cov(singular):
    """
    Calculates cross-block covariance of `singular` values

    Parameters
    ----------
    singular : array (diagonal, k x k)
        singular values from SVD

    Returns
    -------
    array : cross-block covariance (k,)
    """

    squared_sing = np.diag(singular)**2

    return squared_sing / squared_sing.sum()


def kaiser_criterion(singular):
    """
    Determines if variance explained by `singular` value > Kaiser criterion

    Kaiser criterion is 1/# singular values. If cross-block covariance
    explained by singular value exceeds criterion, return True; else, return
    False

    Parameters
    ----------
    singular : array (diagonal, k x k)
        singular values from SVD

    Returns
    -------
    array : boolean (k,)
    """

    return crossblock_cov(singular) > (1/len(singular))


def boot_sig(boot):
    """
    Determines which entries of `boot` are significant via CI crossing

    If CI crosses zero, then bootstrap value is not

    Parameters
    ----------
    boot : array (k x 2)
        components x confidence interval

    Returns
    -------
    array : bool (k,)
    """

    return np.sign(boot).sum(axis=1)
