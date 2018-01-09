# -*- coding: utf-8 -*-

from itertools import repeat
import multiprocessing as mp
import numpy as np
from tqdm import tqdm, trange
from pyls import utils


class BasePLS():
    """
    Parameters
    ----------
    n_perm : int, optional
        Number of permutations to generate. Default: 5000
    n_boot : int, optional
        Number of bootstraps to generate. Default: 1000
    n_split : int, optional
        Number of split-half resamples during each permutation. Default: 500
    ci : (0, 100) float, optional
        Confidence interval to calculate from bootstrapped distributions.
        Default: 95
    n_proc : int, optional
        Number of processors to use for permutation and bootstrapping.
        Default: 1 (no multiprocessing)
    seed : int, optional
        Seed for random number generator. Default: None

    References
    ----------
    .. [1] McIntosh, A. R., Bookstein, F. L., Haxby, J. V., & Grady, C. L.
       (1996). Spatial pattern analysis of functional brain images using
       partial least squares. Neuroimage, 3(3), 143-157.
    .. [2] McIntosh, A. R., & Lobaugh, N. J. (2004). Partial least squares
       analysis of neuroimaging data: applications and advances. Neuroimage,
       23, S250-S263.
    .. [3] Krishnan, A., Williams, L. J., McIntosh, A. R., & Abdi, H. (2011).
       Partial Least Squares (PLS) methods for neuroimaging: a tutorial and
       review. Neuroimage, 56(2), 455-475.
    .. [4] Kovacevic, N., Abdi, H., Beaton, D., & McIntosh, A. R. (2013).
       Revisiting PLS resampling: comparing significance versus reliability
       across range of simulations. In New Perspectives in Partial Least
       Squares and Related Methods (pp. 159-170). Springer, New York, NY.
       Chicago
    """

    def __init__(self, n_perm=5000, n_boot=1000, n_split=500,
                 ci=95, n_proc=1, seed=None):
        self.n_perm, self.n_boot, self.n_split = n_perm, n_boot, n_split
        self.ci = ci
        self._n_proc = n_proc
        self._rs = utils.get_seed(seed)

    def _svd(self):
        """Should compute SVD of cross-covariance matrix of input data."""

        raise NotImplementedError()

    def _run_svd(self):
        """Should run self._svd() on input data."""

        raise NotImplementedError()

    def _split_half(self, X, Y, grouping=None, n_split=500, seed=None):
        """
        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects, ``K`` is the
            number of variables, and ``G`` is a grouping factor (if there are
            multiple groups)
        Y : (N x J) array_like
            Input array, where ``N`` is the number of subjects, ``J`` is the
            number of variables, and ``G`` is a grouping factor (if there are
            multiple groups)
        grouping : (N,) array_like, optional
            Grouping array, where ``len(np.unique(grouping))`` is the number of
            distinct groups in ``X`` and ``Y``. Cross-covariance matrices are
            computed separately for each group and are stacked row-wise.
        n_split : int, optional
            Number of split-half resamples during permutation testing.
            Default: 100
        seed : {int, RandomState instance, None}, optional
            The seed of the pseudo random number generator to use when
            shuffling the data.  If int, ``seed`` is the seed used by the
            random number generator. If RandomState instance, ``seed`` is the
            random number generator. If None, the random number generator is
            the RandomState instance used by ``np.random``. Default: None

        Returns
        -------
        ucorr : (L,) np.ndarray
            Average correlation of left singular vectors across split-halves
        vcorr : (L,) np.ndarray
            Average correlation of right singular vectors across split-halves
        """

        # RandomState generator
        rs = utils.get_seed(seed)

        # original SVD for use in later projection
        U, d, V = self._svd(X, Y, grouping, seed=rs)
        di = np.linalg.inv(d)
        vd, ud = V @ di, U @ di

        # empty arrays to hold split-half correlations
        ucorr = np.zeros((n_split, U.shape[-1]))
        vcorr = np.zeros((n_split, V.shape[-1]))

        for n in range(n_split):
            # empty array to determine split halves
            split = np.zeros(len(X), dtype='bool')
            # get indices for splits, respecting groups if needed
            if grouping is not None:
                for n, grp in enumerate(np.unique(grouping)):
                    take = [np.ceil, np.floor][n % 2]
                    curr_group = grouping == grp
                    inds = rs.choice(np.argwhere(curr_group).squeeze(),
                                     size=int(take(np.sum(curr_group)/2)),
                                     replace=False)
                    split[inds] = True
                D1 = utils.xcorr(X[split], Y[split], grouping[split])
                D2 = utils.xcorr(X[~split], Y[~split], grouping[~split])
            else:
                inds = rs.choice(len(X), size=len(X)//2, replace=False)
                split[inds] = True
                D1 = utils.xcorr(X[split], Y[split])
                D2 = utils.xcorr(X[~split], Y[~split])

            # project cross-covariance matrices onto original SVD to obtain
            # left & right singular vector
            U1, U2 = D1 @ vd, D2 @ vd
            V1, V2 = D1.T @ ud, D2.T @ ud

            # correlate all the singular vectors between split halves
            ucorr[n] = [np.corrcoef(u1, u2)[0, 1] for (u1, u2) in
                        zip(U1.T, U2.T)]
            vcorr[n] = [np.corrcoef(v1, v2)[0, 1] for (v1, v2) in
                        zip(V1.T, V2.T)]

        # return average correlations for singular vectors across ``n_splits``
        return ucorr.mean(axis=0), vcorr.mean(axis=0)

    def _bootstrap(self, X, Y, grouping=None, n_boot=500):
        """
        Bootstraps ``X`` and ``Y`` (w/replace) and computes SE of sing values

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        grouping : (N,) array_like, optional
        n_boot : int, optional
            Number of boostraps to run. Default: 500

        Returns
        -------
        (J[*G] x L x B) ndarray
            Left singular vectors, where ``B = n_boot``
        (K x L x B) ndarray
            Right singular vectors, where ``B = n_boot``
        """

        # "original_u", "original_v" from Matlab PLS
        U_orig, d_orig, V_orig = self._svd(X, Y, grouping=grouping,
                                           seed=self._rs)
        U_boot = np.zeros(U_orig.shape + (n_boot,))
        V_boot = np.zeros(V_orig.shape + (n_boot,))
        bootsamp = np.zeros((len(X), n_boot))

        for i in trange(n_boot, desc='Bootstraps'):
            if grouping is not None:
                inds = np.zeros(len(grouping))
                for grp in np.unique(grouping):
                    curr_group = np.argwhere(grouping == grp).squeeze()
                    inds[curr_group] = self._rs.choice(curr_group,
                                                       size=len(curr_group),
                                                       replace=True)
            else:
                inds = self._rs.choice(np.arange(len(X)), size=len(X),
                                       replace=True)
            U, d_boot, V = self._svd(X[inds], Y[inds], grouping=grouping,
                                     seed=self._rs)

            U_boot[:, :, i], Q = utils.procrustes(U_orig, U, d_boot)
            V_boot[:, :, i] = V @ Q
            bootsamp[:, i] = inds

        return U_boot, V_boot, bootsamp

    def _permutation(self, X, Y, grouping=None,
                     n_perm=1000, n_split=None, n_proc=1):
        """
        Parallelizes ``single_perm()`` to ``n_procs``

        Uses ``starmap_async`` with ``multiprocessing.Pool()`` to parallelize
        jobs. Each job will get a unique random seed to avoid re-use.

        Parameters
        ----------
        X : (N x K [x G]) array_like
        Y : (N x J [x G]) array_like
        grouping : (N,) array_like, optional
            Grouping array, where ``len(np.unique(grouping))`` is the number of
            distinct groups in ``X`` and ``Y``. Default: None
        n_perm : int, optional
            Number of permutations to run. Default: 1000
        n_split : int, optional
            Number of split-half resamples to run. Default: None
        n_proc : int, optional
            Number of processes to use. Default: 1 (no multiprocessing)

        Returns
        -------
        ndarray
            Distributions of singular values
        """

        def callback(result):
            permuted_values.append(result)

        permuted_values = []
        seeds = self._rs.choice(100000, n_perm, replace=False)

        if n_proc > 1:
            pool = mp.Pool(n_proc)
            tqdm(pool.starmap_async(self._single_perm,
                                    zip(repeat(X), repeat(Y), repeat(grouping),
                                        repeat(n_split), seeds),
                                    callback=callback),
                 total=n_perm)
            pool.close()
            pool.join()
        else:
            for n in trange(n_perm, desc='Permutations'):
                permuted_values.append(self._single_perm(X, Y,
                                                         grouping=grouping,
                                                         n_split=n_split,
                                                         seed=seeds[n]))

        permuted_values = np.asarray(permuted_values)

        if n_split is not None:
            permuted_values = permuted_values.transpose(0, 2, 1)

        return permuted_values

    def _single_perm(self, X, Y, grouping=None, n_split=None, seed=None):
        """
        Permutes ``X`` (w/o replacement) and computes SVD of cross-corr matrix

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        grouping : (N,) array_like, optional
            Grouping array, where ``len(np.unique(grouping))`` is the number of
            distinct groups in ``X`` and ``Y``. Default: None
        n_split : int, optional
            Number of split-half resamples to run. Default: None
        seed : int, optional
            Whether to set random seed for reproducibility. Default: None

        Returns
        -------
        ndarray
            Sum of squared, permuted singular values
        """

        rs = utils.get_seed(seed)

        if grouping is not None:
            X_perm = np.row_stack([rs.permutation(X[grouping == grp])
                                   for grp in np.unique(grouping)])
        else:
            X_perm = rs.permutation(X)

        if n_split is not None:
            ucorr, vcorr = self._split_half(X_perm, Y,
                                            grouping=grouping,
                                            n_split=n_split,
                                            seed=rs)
            return ucorr, vcorr

        U, d, V = self._svd(X_perm, Y, grouping=grouping, seed=rs)

        return np.sqrt((d**2).sum(axis=0))

    def _run_perms(self):
        """
        """

        perms = self._permutation(self.X, self.Y,
                                  grouping=self.groups,
                                  n_perm=self.n_perm,
                                  n_split=self.n_split,
                                  n_proc=self._n_proc)

        if self.n_split is not None:
            self.u_pvals = utils.perm_sig(perms[:, :, 0], np.diag(self.ucorr))
            self.v_pvals = utils.perm_sig(perms[:, :, 1], np.diag(self.vcorr))
        else:
            self.d_pvals = utils.perm_sig(perms, self.d)

    def _run_boots(self):
        """
        """

        U_boot, V_boot = self._bootstrap(self.X, self.Y,
                                         self.U, self.V,
                                         grouping=self.groups,
                                         n_boot=self.n_boot)
        self.U_bci, self.V_bci = utils.boot_ci(U_boot, V_boot, ci=self.ci)
        self.U_bsr, self.V_bsr = utils.boot_rel(self.U, self.V, U_boot, V_boot)

    def _get_sig(self):
        """
        Determines the significance of returned singular vectors using the
        Kaiser criterion, crossblock covariance, and whether the bootstrap
        confidence interval crosses zero (boolean).
        """

        self.U_sig = utils.boot_sig(self.U_bci)
        self.V_sig = utils.boot_sig(self.V_bci)

        self.d_kaiser = utils.kaiser_criterion(self.d)
        self.d_varexp = utils.crossblock_cov(self.d)
