# -*- coding: utf-8 -*-

import warnings
import numpy as np
from sklearn.utils.extmath import randomized_svd
from pyls import compute, utils


class PLSInputs():
    """
    Class to hold PLS input information

    Parameters
    ----------
    groups : (N,) array_like, optional
        Array with labels separating ``N`` subjects into ``G`` groups. Default:
        None
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

    def __init__(self, X=None, Y=None, groups=None, n_cond=1,
                 n_perm=5000, n_boot=1000, n_split=500,
                 ci=95, n_proc=1, seed=None):
        # important inputs
        self._X, self._Y = X, Y
        self._groups, self._n_cond = groups, n_cond
        self._n_perm, self._n_boot, self._n_split = n_perm, n_boot, n_split
        self._ci = ci
        self._n_proc = n_proc
        self._seed = seed

    @property
    def n_cond(self):
        """Number of conditions"""
        return self._n_cond

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
    Base PLS class

    Implements most of the math required for PLS, leaving a few functions
    for PLS sub-classes to implement.

    Parameters
    ----------
    groups : (G,) list
        List with number of subjects in each of ``G`` groups.
    n_cond : int, optional
        Number of conditions. Default: 1
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

    def __init__(self, X, Y=None, groups=None, n_cond=1,
                 n_perm=5000, n_boot=1000, n_split=500,
                 ci=95, n_proc=1, seed=None):
        # if groups aren't provided but conditions are, use groups instead
        # otherwise, just get number of subjects
        if groups is None:
            groups = [len(X)]
        if len(groups) == 1 and n_cond > 1:
            groups, n_cond = [len(X) // n_cond] * n_cond, 1
        self.inputs = PLSInputs(X=X, Y=Y,
                                groups=groups, n_cond=n_cond,
                                n_perm=n_perm, n_boot=n_boot, n_split=n_split,
                                ci=ci, n_proc=n_proc, seed=seed)
        self._rs = utils.get_seed(self.inputs.seed)

    def _run_pls(self, *args, **kwargs):
        """
        Runs entire PLS analysis
        """

        raise NotImplementedError

    def _gen_covcorr(self, X, Y, groups):
        """
        Generates cross-covariance array to be used in ``self._svd()``

        Parameters
        ----------
        X : (N x M) array_like
        Y : (N x F) array_like
        groups : (N x Y) array_like, optional

        Returns
        -------
        crosscov : np.ndarray
            Covariance array for decomposition
        """

        raise NotImplementedError

    def _gen_permsamp(self):
        """
        Generates permutation arrays to be used in ``self._permutation()``
        """

        Y = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        permsamp = np.zeros(shape=(len(Y), self.inputs.n_perm), dtype=int)
        subj_inds = np.arange(np.sum(self.inputs.groups), dtype=int)
        warned = False

        # calculate some variables for permuting conditions within subject
        # do this here to save on calculation time
        indices, grps = np.where(Y)
        grp_conds = np.split(indices, np.where(np.diff(grps))[0] + 1)
        to_permute = [np.vstack(grp_conds[i:i + self.inputs.n_cond]) for i in
                      range(0, Y.shape[-1], self.inputs.n_cond)]
        splitinds = np.cumsum(self.inputs.groups)[:-1]
        check_grps = utils.dummy_code(self.inputs.groups).T.astype(bool)

        for i in utils.trange(self.inputs.n_perm, desc='Making permutations'):
            count, duplicated = 0, True
            while duplicated and count < 500:
                count, duplicated = count + 1, False
                # generate conditions permuted w/i subject
                inds = np.hstack([utils.permute_cols(i, seed=self._rs) for i
                                  in to_permute])
                # generate permutation of subjects across groups
                perm = self._rs.permutation(subj_inds)
                # confirm subjects *are* mixed across groups
                if len(self.inputs.groups) > 1:
                    for grp in check_grps:
                        if np.all(np.sort(perm[grp]) == subj_inds[grp]):
                            duplicated = True
                # permute conditions w/i subjects across groups and stack
                perminds = np.hstack([f.flatten('F') for f in
                                      np.split(inds[:, perm].T, splitinds)])
                # make sure permuted indices are not a duplicate sequence
                dupe_seq = perminds[:, None] == permsamp[:, :i]
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            # if we broke out because we tried 500 permutations and couldn't
            # generate a new one, just warn that we're using duplicate
            # permutations and give up
            if count == 500 and not warned:
                warnings.warn('WARNING: Duplicate permutations used.')
                warned = True
            # store the permuted indices
            permsamp[:, i] = perminds

        return permsamp

    def _gen_bootsamp(self):
        """
        Generates bootstrap arrays to be used in ``self._bootstrap()``
        """

        Y = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        bootsamp = np.zeros(shape=(len(Y), self.inputs.n_boot), dtype=int)
        min_subj = int(np.ceil(Y.sum(axis=0).min() * 0.5))
        subj_inds = np.arange(np.sum(self.inputs.groups), dtype=int)
        warned = False

        # calculate some variables for ensuring we resample with replacement
        # subjects across all their conditions. do this here to save on
        # calculation time
        indices, grps = np.where(Y)
        grp_conds = np.split(indices, np.where(np.diff(grps))[0] + 1)
        inds = np.hstack([np.vstack(grp_conds[i:i + self.inputs.n_cond]) for i
                          in range(0, len(grp_conds), self.inputs.n_cond)])
        splitinds = np.cumsum(self.inputs.groups)[:-1]
        check_grps = utils.dummy_code(self.inputs.groups).T.astype(bool)

        for i in utils.trange(self.inputs.n_boot, desc='Making bootstraps'):
            count, duplicated = 0, True
            while duplicated and count < 500:
                count, duplicated = count + 1, False
                # empty container to store current bootstrap attempt
                boot = np.zeros(shape=(subj_inds.size), dtype=int)
                # iterate through and resample from w/i groups
                for grp in check_grps:
                    curr_grp, all_same = subj_inds[grp], True
                    while all_same:
                        boot[curr_grp] = self._rs.choice(curr_grp,
                                                         size=curr_grp.size,
                                                         replace=True)
                        # make sure bootstrap has enough unique subjs
                        if np.unique(boot[curr_grp]).size >= min_subj:
                            all_same = False
                # resample subjects (with conditions) and stack groups
                bootinds = np.hstack([f.flatten('F') for f in
                                      np.split(inds[:, boot].T, splitinds)])
                # make sure bootstrap is not a duplicated sequence
                dupe_seq = (np.sort(bootinds[:, None], axis=0) ==
                            np.sort(bootsamp[:, :i], axis=0))
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            # if we broke out because we tried 500 bootstraps and couldn't
            # generate a new one, just warn that we're using duplicate
            # bootstraps and give up
            if count == 500 and not warned:
                warnings.warn('WARNING: Duplicate bootstraps used.')
                warned = True
            # store the bootstrapped indices
            bootsamp[:, i] = bootinds

        return bootsamp

    def _gen_splits(self):
        """
        Generates split-half arrays to be used in ``self._split_half()``
        """

        Y = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        splitsamp = np.zeros(shape=(len(Y), self.inputs.n_split), dtype=bool)
        subj_inds = np.arange(np.sum(self.inputs.groups), dtype=int)
        warned = False

        # calculate some variables for permuting conditions within subject
        # do this here to save on calculation time
        indices, grps = np.where(Y)
        grp_conds = np.split(indices, np.where(np.diff(grps))[0] + 1)
        inds = np.hstack([np.vstack(grp_conds[i:i + self.inputs.n_cond]) for i
                          in range(0, len(grp_conds), self.inputs.n_cond)])
        splitinds = np.cumsum(self.inputs.groups)[:-1]
        check_grps = utils.dummy_code(self.inputs.groups).T.astype(bool)

        for i in range(self.inputs.n_split):
            count, duplicated = 0, True
            while duplicated and count < 500:
                count, duplicated = count + 1, False
                # empty containter to store current split half attempt
                split = np.zeros(shape=(subj_inds.size), dtype=bool)
                # iterate through and split each group separately
                for grp in check_grps:
                    curr_grp = subj_inds[grp]
                    take = self._rs.choice([np.ceil, np.floor])
                    num_subj = int(take(curr_grp.size/2))
                    splinds = self._rs.choice(curr_grp,
                                              size=num_subj,
                                              replace=False)
                    split[splinds] = True
                # split subjects (with conditions) and stack groups
                half = np.hstack([f.flatten('F') for f in
                                  np.split((inds.astype(bool) * split[None]).T,
                                           splitinds)])
                # make sure split half is not a duplicated sequence
                dupe_seq = half[:, None] == splitsamp[:, :i]
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500 and not warned:
                warnings.warn('WARNING: Duplicate split halves used.')
                warned = True
            splitsamp[:, i] = half

        return splitsamp

    def _svd(self, X, Y, seed=None):
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

        Returns
        -------
        U : (J x J-1) ndarray
            Left singular vectors
        d : (J-1 x J-1) ndarray
            Diagonal array of singular values
        V : (K x J-1) ndarray
            Right singular vectors
        """

        crosscov = self._gen_covcorr(X, Y,
                                     utils.dummy_code(self.inputs.groups,
                                                      self.inputs.n_cond))
        U, d, V = randomized_svd(crosscov,
                                 n_components=Y.shape[-1]-1,
                                 random_state=utils.get_seed(seed))

        return U, np.diag(d), V.T

    def _bootstrap(self, X, Y):
        """
        Bootstraps ``X`` and ``Y`` (w/replacement) and recomputes SVD

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like

        Returns
        -------
        U_boot : (J[*G] x L x B) np.ndarray
            Left singular vectors
        V_boot : (K x L x B) np.ndarray
            Right singular vectors
        """

        # generate bootstrap resampled indices
        self.bootsamp = self._gen_bootsamp()

        # get original values
        U_orig, d_orig, V_orig = self._svd(X, Y, seed=self._rs)
        U_boot = np.zeros(shape=U_orig.shape + (self.inputs.n_boot,))
        V_boot = np.zeros(shape=V_orig.shape + (self.inputs.n_boot,))

        for i in utils.trange(self.inputs.n_boot, desc='Running bootstraps'):
            inds = self.bootsamp[:, i]
            U, d, V = self._svd(X[inds], Y[inds], seed=self._rs)
            U_boot[:, :, i], rotate = compute.procrustes(U_orig, U, d)
            V_boot[:, :, i] = V @ d @ rotate

        return U_boot, V_boot

    def _permutation(self, X, Y):
        """
        Permutes ``X`` and ``Y`` (w/o replacement) and recomputes SVD

        Parameters
        ----------
        X : (N x K [x G]) array_like
        Y : (N x J [x G]) array_like

        Returns
        -------
        d_perm : (L x P) np.ndarray
            Permuted singular values, where ``L`` is the number of singular
            values and ``P`` is the number of permutations
        ucorrs : (L x P) np.ndarray
            Split-half correlations of left singular values. Only useful if
            ``self.inputs.n_split != 0``
        vcorrs : (L x P) np.ndarray
            Split-half correlations of right singular values. Only useful if
            ``self.inputs.n_split != 0``
        """

        # generate permuted indices
        self.permsamp = self._gen_permsamp()

        # get original values
        U_orig, d_orig, V_orig = self._svd(X, Y, seed=self._rs)

        d_perm = np.zeros(shape=(len(d_orig), self.inputs.n_perm))
        ucorrs = np.zeros(shape=(len(d_orig), self.inputs.n_perm))
        vcorrs = np.zeros(shape=(len(d_orig), self.inputs.n_perm))

        for i in utils.trange(self.inputs.n_perm, desc='Running permutations'):
            inds = self.permsamp[:, i]
            outputs = self._single_perm(X[inds], Y)
            d_perm[:, i] = outputs[0]
            if self.inputs.n_split is not None:
                ucorrs[:, i], vcorrs[:, i] = outputs[1:]

        return d_perm, ucorrs, vcorrs

    def _single_perm(self, X, Y):
        """
        Permutes ``X`` (w/o replacement) and computes SVD of cross-corr matrix

        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like

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
        U, d, V = self._svd(X, Y, seed=self._rs)
        ssd = np.sqrt((d**2).sum(axis=0))

        # get ucorr/vcorr if split-half resampling requested
        if self.inputs.n_split is not None:
            di = np.linalg.inv(d)
            ud, vd = U @ di, V @ di
            ucorr, vcorr = self._split_half(X, Y, ud, vd)
        else:
            ucorr, vcorr = None, None

        return ssd, ucorr, vcorr

    def _split_half(self, X, Y, ud, vd):
        """
        Parameters
        ----------
        X : (N x K) array_like
        Y : (N x J) array_like
        ud : (K[*G] x L) array_like
        vd : (J x L) array_like

        Returns
        -------
        ucorr : (L,) np.ndarray
            Average correlation of left singular vectors across split-halves
        vcorr : (L,) np.ndarray
            Average correlation of right singular vectors across split-halves
        """

        # generate splits
        splitsamp = self._gen_splits()

        # empty arrays to hold split-half correlations
        ucorr = np.zeros(shape=(ud.shape[-1], self.inputs.n_split))
        vcorr = np.zeros(shape=(vd.shape[-1], self.inputs.n_split))

        for i in range(self.inputs.n_split):
            spl = splitsamp[:, i]
            D1 = self._gen_covcorr(X[spl], Y[spl],
                                   utils.dummy_code(self.inputs.groups,
                                                    self.inputs.n_cond)[spl])
            D2 = self._gen_covcorr(X[~spl], Y[~spl],
                                   utils.dummy_code(self.inputs.groups,
                                                    self.inputs.n_cond)[~spl])

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
