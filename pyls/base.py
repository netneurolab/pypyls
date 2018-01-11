# -*- coding: utf-8 -*-

import numpy as np
from pyls import compute, utils


class PLSInputs():
    """
    Class to hold PLS input information
    """

    def __init__(self, n_perm=5000, n_boot=1000, n_split=None,
                 ci=95, n_proc=1, seed=None, verbose=False):
        self._n_perm, self._n_boot, self._n_split = n_perm, n_boot, n_split
        self._ci = ci
        self._n_proc = n_proc
        self._verbose = verbose
        self._seed = seed
        # to be set at a later time and place
        self._X, self._Y, self._grouping = None, None, None

    @property
    def n_perm(self):
        """Number of permutations"""
        return self._n_perm

    @property
    def n_boot(self):
        """Number of bootstraps"""
        return self._n_boot

    @property
    def n_split(self):
        """Number of split-half resamples"""
        return self._n_split

    @property
    def ci(self):
        """Requested confidence interval for bootstrap testing"""
        return self._ci

    @property
    def n_proc(self):
        """Number of processors (for multiprocessing)"""
        return self._n_proc

    @property
    def seed(self):
        """Provided pseudo random seed"""
        return self._seed

    @property
    def X(self):
        """Provided ``X`` data matrix"""
        return self._X

    @property
    def Y(self):
        """Provided ``Y`` data matrix"""
        return self._Y

    @property
    def grouping(self):
        """Provided group labels"""
        return self._grouping


class BasePLS():
    """
    Parameters
    ----------
    n_perm : int, optional
        Number of permutations to generate. Default: 5000
    n_boot : int, optional
        Number of bootstraps to generate. Default: 1000
    n_split : int, optional
        Number of split-half resamples during each permutation. Default: None
    ci : (0, 100) float, optional
        Confidence interval to calculate from bootstrapped distributions.
        Default: 95
    n_proc : int, optional
        Number of processors to use for permutation and bootstrapping.
        Default: 1 (no multiprocessing)
    seed : int, optional
        Seed for random number generator. Default: None
    verbose : bool, optional
        Print status updates

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

    def __init__(self, n_perm=5000, n_boot=1000, n_split=None,
                 ci=95, n_proc=1, seed=None, verbose=False):
        self.inputs = PLSInputs(n_perm=n_perm,
                                n_boot=n_boot,
                                n_split=n_split,
                                ci=ci,
                                n_proc=n_proc,
                                seed=seed,
                                verbose=verbose)
        self._rs = utils.get_seed(self.inputs.seed)

    def _run_pls(self, *args, **kwargs):
        """
        Should run entire PLS analysis
        """

        raise NotImplementedError

    def _svd(self, *args, **kwargs):
        """
        Should compute SVD of cross-covariance matrix of input data
        """

        raise NotImplementedError

    def _gen_permsamp(self, *args, **kwargs):
        """
        Generates permutation arrays to be using in ``self._permutation()``
        """

        raise NotImplementedError

    def _gen_bootsamp(self, *args, **kwargs):
        """
        Generates bootstrap arrays to be used in ``self._bootstrap()``
        """

        raise NotImplementedError

    def _gen_splits(self, *args, **kwargs):
        """
        Generates split half arrays to be using in ``self._split_half()``
        """

        raise NotImplementedError

    def _bootstrap(self, X, Y, grouping=None):
        """
        Bootstraps ``X`` and ``Y`` (w/replacement) and recomputes SVD

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        grouping : (N,) array_like, optional

        Returns
        -------
        (J[*G] x L x B) ndarray
            Left singular vectors
        (K x L x B) ndarray
            Right singular vectors
        """

        # generate bootstrap resampled indices
        self.bootsamp = self._gen_bootsamp(X, Y, grouping=grouping)

        # "original_u", "original_v" from Matlab PLS
        U_orig, d_orig, V_orig = self._svd(X, Y, grouping=grouping,
                                           seed=self._rs)
        U_boot = np.zeros(shape=U_orig.shape + (self.inputs.n_boot,))
        V_boot = np.zeros(shape=V_orig.shape + (self.inputs.n_boot,))

        for i in utils.trange(self.inputs.n_boot, desc='Running bootstraps'):
            inds = self.bootsamp[:, i]
            U, d, V = self._svd(X[inds], Y[inds], grouping=grouping,
                                seed=self._rs)
            U_boot[:, :, i], Q = compute.procrustes(U_orig, U, d)
            V_boot[:, :, i] = V @ d @ Q

        return U_boot, V_boot

    def _permutation(self, X, Y, grouping=None):
        """
        Permutes ``X`` and ``Y`` (w/o replacement) and recomputes SVD

        Parameters
        ----------
        X : (N x K [x G]) array_like
        Y : (N x J [x G]) array_like
        grouping : (N,) array_like, optional
            Grouping array, where ``len(np.unique(grouping))`` is the number of
            distinct groups in ``X`` and ``Y``. Default: None

        Returns
        -------
        permuted_values : np.ndarray
            Distributions of singular values
        """

        def callback(result):
            permuted_values.append(result)

        self.permsamp = self._gen_permsamp(X, Y, grouping=grouping)
        seeds = self._rs.choice(100000, self.inputs.n_perm, replace=False)

        permuted_values = []
        for i in utils.trange(self.inputs.n_perm, desc='Running permutations'):
            permuted_values.append(self._single_perm(X[self.permsamp[:, i]], Y,
                                                     grouping=grouping,
                                                     seed=seeds[i]))

        permuted_values = np.asarray(permuted_values)

        if self.inputs.n_split is not None:
            permuted_values = permuted_values.transpose(0, 2, 1)

        return permuted_values

    def _single_perm(self, X, Y, grouping=None, seed=None):
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

        if self.inputs.n_split is not None:
            ucorr, vcorr = self._split_half(X, Y,
                                            grouping=grouping,
                                            seed=rs)
            return ucorr, vcorr

        U, d, V = self._svd(X, Y, grouping=grouping, seed=rs)

        return np.sqrt((d**2).sum(axis=0))

    def _split_half(self, X, Y, grouping=None, seed=None):
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
        U, d, V = self._svd(X, Y, grouping=grouping, seed=rs)
        di = np.linalg.inv(d)
        vd, ud = V @ di, U @ di

        # empty arrays to hold split-half correlations
        ucorr = np.zeros((self.inputs.n_split, U.shape[-1]))
        vcorr = np.zeros((self.inputs.n_split, V.shape[-1]))

        for n in range(self.inputs.n_split):
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
