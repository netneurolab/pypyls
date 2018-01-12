# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils.extmath import randomized_svd
from pyls import compute, utils


class PLSInputs():
    """
    Class to hold PLS input information

    Parameters
    ----------
    n_perm : int, optional
        Number of permutations for testing statistical significance of singular
        vectors. Default: 5000
    n_boot : int, optional
        Number of bootstraps for testing reliability of singular vectors.
        Default: 1000
    n_split : int, optional
        Number of split-half resamples for testing reliability of permutations.
        Default: 500
    ci : (0, 100) float, optional
        Confidence interval used to calculate reliability of features across
        bootstraps. This value approximately corresponds to setting an alpha
        value, where ``alpha = (100 - ci) / 100``. Default: 95
    n_proc : int, optional
        Number of processors to use for permutation and bootstrapping.
        Default: 1 (no multiprocessing)
    seed : int, optional
        Seed for random number generator. Default: None
    """

    def __init__(self, n_perm=5000, n_boot=1000, n_split=500,
                 ci=95, n_proc=1, seed=None):
        self._n_perm, self._n_boot, self._n_split = n_perm, n_boot, n_split
        self._ci = ci
        self._n_proc = n_proc
        self._seed = seed
        # to be set at a later time and place
        self._X, self._Y, self._groups = None, None, None

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
        """Number of processors requested (for multiprocessing)"""
        return self._n_proc

    @property
    def seed(self):
        """Pseudo random seed"""
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
    def groups(self):
        """Provided group labels"""
        return self._groups


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
        self.inputs = PLSInputs(n_perm=n_perm,
                                n_boot=n_boot,
                                n_split=n_split,
                                ci=ci,
                                n_proc=n_proc,
                                seed=seed)
        self._rs = utils.get_seed(self.inputs.seed)

    def _run_pls(self, *args, **kwargs):
        """
        Runs entire PLS analysis
        """

        raise NotImplementedError

    def _gen_covcorr(self, *args, **kwargs):
        """
        Generates cross-covariance array to be used in ``self._svd()``
        """

        raise NotImplementedError

    def _gen_permsamp(self, *args, **kwargs):
        """
        Generates permutation arrays to be used in ``self._permutation()``
        """

        raise NotImplementedError

    def _gen_bootsamp(self, *args, **kwargs):
        """
        Generates bootstrap arrays to be used in ``self._bootstrap()``
        """

        raise NotImplementedError

    def _gen_splits(self, *args, **kwargs):
        """
        Generates split-half arrays to be used in ``self._split_half()``
        """

        raise NotImplementedError

    def _svd(self, X, Y, seed=None, groups=None):
        """
        Runs SVD on cross-covariance matrix computed from ``X`` and ``Y``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Input array, where ``N`` is the number of subjects and ``J`` is The
            number of variables
        groups : placeholder
            Grouping array, where ``len(np.unique(groups))`` is the number of
            distinct groups in ``X`` and ``Y``. Default: None

        Returns
        -------
        U : (J x J-1) ndarray
            Left singular vectors
        d : (J-1 x J-1) ndarray
            Diagonal array of singular values
        V : (K x J-1) ndarray
            Right singular vectors
        """

        crosscov = self._gen_covcorr(X, Y, groups=groups)
        U, d, V = randomized_svd(crosscov,
                                 n_components=Y.shape[-1]-1,
                                 random_state=utils.get_seed(seed))

        return U, np.diag(d), V.T

    def _bootstrap(self, X, Y, groups=None):
        """
        Bootstraps ``X`` and ``Y`` (w/replacement) and recomputes SVD

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        groups : (N,) array_like, optional
            Grouping array, where ``len(np.unique(groups))`` is the number of
            distinct groups in ``X`` and ``Y``. Default: None

        Returns
        -------
        U_boot : (J[*G] x L x B) np.ndarray
            Left singular vectors
        V_boot : (K x L x B) np.ndarray
            Right singular vectors
        """

        # generate bootstrap resampled indices
        self.bootsamp = self._gen_bootsamp(X, Y, groups=groups)

        # get original values
        U_orig, d_orig, V_orig = self._svd(X, Y, groups=groups,
                                           seed=self._rs)
        U_boot = np.zeros(shape=U_orig.shape + (self.inputs.n_boot,))
        V_boot = np.zeros(shape=V_orig.shape + (self.inputs.n_boot,))

        for i in utils.trange(self.inputs.n_boot, desc='Running bootstraps'):
            inds = self.bootsamp[:, i]
            U, d, V = self._svd(X[inds], Y[inds], groups=groups,
                                seed=self._rs)
            U_boot[:, :, i], rotate = compute.procrustes(U_orig, U, d)
            V_boot[:, :, i] = V @ d @ rotate

        return U_boot, V_boot

    def _permutation(self, X, Y, groups=None):
        """
        Permutes ``X`` and ``Y`` (w/o replacement) and recomputes SVD

        Parameters
        ----------
        X : (N x K [x G]) array_like
        Y : (N x J [x G]) array_like
        groups : (N,) array_like, optional
            Grouping array, where ``len(np.unique(groups))`` is the number of
            distinct groups in ``X`` and ``Y``. Default: None

        Returns
        -------
        d_perm : (L x P) np.ndarray
            Permuted singular values, where ``L`` is the number of singular
            values and ``P`` is the number of permutations
        ucorrs : (L x P) np.ndarray
            Split-half correlations of left singular values. Only useful if
            ``n_split != 0``
        vcorrs : (L x P) np.ndarray
            Split-half correlations of right singular values. Only useful if
            ``n_split != 0``
        """

        # generate permuted indices
        self.permsamp = self._gen_permsamp(X, Y, groups=groups)

        # get original values
        U_orig, d_orig, V_orig = self._svd(X, Y, groups=groups,
                                           seed=self._rs)

        d_perm = np.zeros(shape=(len(d_orig), self.inputs.n_perm))
        ucorrs = np.zeros(shape=(len(d_orig), self.inputs.n_perm))
        vcorrs = np.zeros(shape=(len(d_orig), self.inputs.n_perm))

        for i in utils.trange(self.inputs.n_perm, desc='Running permutations'):
            inds = self.permsamp[:, i]
            outputs = self._single_perm(X[inds], Y, groups=groups)
            d_perm[:, i] = outputs[0]
            if self.inputs.n_split is not None:
                ucorrs[:, i], vcorrs[:, i] = outputs[1:]

        return d_perm, ucorrs, vcorrs

    def _single_perm(self, X, Y, groups=None):
        """
        Permutes ``X`` (w/o replacement) and computes SVD of cross-corr matrix

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        groups : (N,) array_like, optional
            Grouping array, where ``len(np.unique(groups))`` is the number of
            distinct groups in ``X`` and ``Y``. Default: None

        Returns
        -------
        ssd : (L,) np.ndarray
            Sum of squared, permuted singular values
        ucorr : (L,) np.ndarray
            Split-half correlations of left singular values. Only useful if
            ``n_split != 0``
        vcorr : (L,) np.ndarray
            Split-half correlations of right singular values. Only useful if
            ``n_split != 0``
        """

        # perform SVD of permuted array and get sum of squared singular values
        U, d, V = self._svd(X, Y, groups=groups, seed=self._rs)
        ssd = np.sqrt((d**2).sum(axis=0))

        # get ucorr/vcorr if split-half resampling requested
        if self.inputs.n_split is not None:
            di = np.linalg.inv(d)
            ud, vd = U @ di, V @ di
            ucorr, vcorr = self._split_half(X, Y, ud, vd, groups=groups)
        else:
            ucorr, vcorr = None, None

        return ssd, ucorr, vcorr

    def _split_half(self, X, Y, ud, vd, groups=None):
        """
        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        ud : (K[*G] x L) array_like
        vd : (J x L) array_like
        groups : (N,) array_like, optional
            Grouping array, where ``len(np.unique(groups))`` is the number of
            distinct groups in ``X`` and ``Y``. Default: None

        Returns
        -------
        ucorr : (L,) np.ndarray
            Average correlation of left singular vectors across split-halves
        vcorr : (L,) np.ndarray
            Average correlation of right singular vectors across split-halves
        """

        # generate splits
        splitsamp = self._gen_splits(X, Y, groups=groups)

        # empty arrays to hold split-half correlations
        ucorr = np.zeros(shape=(ud.shape[-1], self.inputs.n_split))
        vcorr = np.zeros(shape=(vd.shape[-1], self.inputs.n_split))

        for i in range(self.inputs.n_split):
            split = splitsamp[:, i]
            if groups is not None:
                D1 = self._gen_covcorr(X[split], Y[split], groups[split])
                D2 = self._gen_covcorr(X[~split], Y[~split], groups[~split])
            else:
                D1 = self._gen_covcorr(X[split], Y[split])
                D2 = self._gen_covcorr(X[~split], Y[~split])

            # project cross-covariance matrices onto original SVD to obtain
            # left & right singular vector
            U1, U2 = D1 @ vd, D2 @ vd
            V1, V2 = D1.T @ ud, D2.T @ ud

            # correlate all the singular vectors between split halves
            ucorr[:, i] = [np.corrcoef(u1, u2)[0, 1] for (u1, u2) in
                           zip(U1.T, U2.T)]
            vcorr[:, i] = [np.corrcoef(v1, v2)[0, 1] for (v1, v2) in
                           zip(V1.T, V2.T)]

        # return average correlations for singular vectors across ``n_split``
        return ucorr.mean(axis=-1), vcorr.mean(axis=-1)
