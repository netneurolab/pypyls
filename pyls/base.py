# -*- coding: utf-8 -*-

import warnings
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_random_state
from pyls import compute, utils


class PLSInputs(utils.DefDict):
    """
    PLS input information

    Attributes
    ----------
    X : (S x B) np.ndarray
        Input data matrix, where ``S`` is observations and ``B`` is features
    Y : (S x T) np.ndarray
        Behavioral matrix, where ``S`` is observations and ``T`` is features
        If from a behavioral PLS, this is the provided behavior matrix; if from
        a mean centered PLS, this is the dummy-coded group/condition matrix.
    groups : (G,) list
        List with number of subjects in each of ``G`` groups
    n_cond : int
        Number of conditions observed in data
    n_perm : int, optional
        Number of permutations generated for testing significance of PLS.
        Default: 5000
    n_boot : int, optional
        Number of bootstraps generated for testing reliability of features.
        Default: 5000
    n_split : int, optional
        Number of split-half resamples for each of ``n_perm`` permutations.
        Default: 0
    cv_split : [0, 1] float, optional
        Proportion of data to partition as test split during cross-validation.
        Default: 0.25
    mean_centering : int, optional
        Mean centering type. Must be in [0, 1, 2]. Default: 0
    rotate : bool, optional
        Whether to perform procrustes rotations during permutations. Can
        inflate false-positive rate (see Kovacevic et al., 2013). Default: True
    ci : float, optional
        Confidence interval to assess bootstrap resampling results. Default: 95
    n_proc : int, optional
        Multiprocessing not implemented yet. Default: 1
    seed : {int, RandomState, None}, optional
        Seed for pseudo-random number generation. Default: None
    """
    defaults = dict(
        X=None, Y=None, groups=None, n_cond=1,
        n_perm=5000, n_boot=5000,
        n_split=100, test_size=0.25,
        mean_centering=None, rotate=True,
        ci=95, n_proc=1, seed=None
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.n_split == 0:
            self.n_split = None
        if self.test_size < 0 or self.test_size >= 1:
            raise ValueError('Test_size must be in [0, 1). Provided value: {}'
                             .format(str(self.test_size)))


class PLSResults(utils.DefDict):
    """
    PLS results information

    Attributes
    ----------
    u : (B x L) np.ndarray
        Left singular vectors from SVD
    s : (L x L) np.ndarray
        Diagonal array of singular values from SVD
    v : (J x L) np.ndarray
        Right singular vectors from SVD
    usc : (S x L) np.ndarray
        Brain scores (``inputs.X @ v``)
    vsc : (S x L) np.ndarray
        Design scores (``inputs.Y @ u``)
    lvcorrs : (J x L) np.ndarray
        Correlation of ``usc`` with ``inputs.Y``
    perm_result : dict-like
        Contains results of permutation testing
    boot_result : dict-like
        Contains results of bootstrap resampling
    perm_splithalf : dict-like
        Contains results of split-half resampling
    inputs : dict-likes
        Contains inputs provided to PLS analysis. See ``pyls.base.PLSInputs``
    """

    class PLSBootResult(utils.DefDict):
        """
        PLS bootstrap resampling results

        Attributes
        ----------
        compare_u : (B x L) np.ndarray
            Left singular vectors normalized by standard errors in ``u_se``.
            Often referred to as "bootstrap ratios" or "BSRs", can usually be
            interpreted as a z-score (assuming a non-skewed distribution)
        u_se : (B x L) np.ndarray
            Standard error of bootstrapped distribution of left singular
            vectors
        distrib : (J x L x R) np.ndarray
            Bootstrapped distribution of either ``orig_usc`` or ``orig_corr``
        usc2 : (S x L) np.ndarray
            Mean-centered brain scores (``(X - mean(X)) @ v``)
        orig_usc : (J x L) np.ndarray
            Group x condition averages of ``usc2``. Can be treated as a
            contrast indicating group x condition differences
        ulusc : (J x L) np.ndarray
            Upper bound of confidence interval for ``orig_usc``
        llusc : (J x L) np.ndarray
            Lower bound of confidence interval for ``orig_usc``
        orig_corr : (J x L) np.ndarray
            Correlation of ``usc`` with ``inputs.Y``
        ulcorr : (J x L) np.ndarray
            Upper bound of confidence interval for ``orig_corr``
        llcorr : (J x L) np.ndarray
            Lower bound of confidence interval for ``orig_corr``
        """
        defaults = dict(
            compare_u=None, u_se=None, distrib=None, usc2=None,
            orig_usc=None, ulusc=None, llusc=None,
            orig_corr=None, ulcorr=None, llcorr=None,
            bootsamp=None
        )

    class PLSPermResult(utils.DefDict):
        """
        PLS permutation results

        Attributes
        ----------
        sp : (L,) np.ndarray
            Number of permutations where singular values exceeded original data
            decomposition for each of ``L`` latent variables
        sprob : (L,) np.ndarray
            ``sp`` normalized by the total number of permutations. Can be
            interpreted as the statistical significance of the latent variables
        """
        defaults = dict(
            sp=None, sprob=None, permsamp=None
        )

    class PLSSplitHalfResult(utils.DefDict):
        """
        PLS split-half resampling results

        Attributes
        ----------
        orig_ucorr, orig_vcorr : (L,) np.ndarray
            Average correlations between split-half resamples in original (non-
            permuted) data for left/right singular vectors. Can be interpreted
            as reliability of ``L`` latent variables
        ucorr_prob, vcorr_prob : (L,) np.ndarray
            Number of permutations where correlation between split-half
            resamples exceeded original correlations, normalized by the total
            number of permutations. Can be interpreted as the statistical
            significance of the reliability of ``L`` latent variables
        ucorr_ul, vcorr_ul : (L,) np.ndarray
            Upper bound of confidence interval for correlations between split
            halves for left/right singular vectors
        ucorr_ll, vcorr_ll : (L,) np.ndarray
            Lower bound of confidence interval for correlations between split
            halves for left/right singular vectors
        """
        defaults = dict(
            orig_ucorr=None, orig_vcorr=None, ucorr_prob=None, vcorr_prob=None,
            ucorr_ul=None, ucorr_ll=None, vcorr_ul=None, vcorr_ll=None
        )

    class PLSCrossValidationResults(utils.DefDict):
        """
        PLS cross-validation results

        Attributes
        ----------
        r_sqared : (T x I) np.ndarray
            R-squared ("determination coefficient") for each of `T` predicted
            behavioral scores against true behavioral scores across `I` train /
            test split
        pearson_r : (T x I) np.ndarray
            Pearson's correlation for each of `T` predicted behavioral scores
            against true behavioral scores across `I` train / test split
        """
        defaults = dict(
            r_squared=None, pearson_r=None
        )

    defaults = dict(
        u=None, s=None, v=None, usc=None, vsc=None, lvcorrs=None, inputs={},
        boot_result={}, perm_result={}, perm_splithalf={}, cross_val={}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = PLSInputs(**self.inputs)
        self.boot_result = self.PLSBootResult(**self.boot_result)
        self.perm_result = self.PLSPermResult(**self.perm_result)
        self.perm_splithalf = self.PLSSplitHalfResult(**self.perm_splithalf)
        self.cross_val = self.PLSCrossValidationResults(**self.cross_val)


class BasePLS():
    """
    Base PLS class

    Implements most of the math required for PLS, leaving a few functions
    for PLS sub-classes to implement.

    Parameters
    ----------
    X : (S x B) array_like
        Input data matrix, where ``S`` is observations and ``B`` is features
    groups : (G,) array_like, optional
        Array with number of subjects in each of ``G`` groups. Default: ``[S]``
    n_cond : int, optional
        Number of conditions. Default: 1
    **kwargs : optional
        See ``pyls.base.PLSInputs`` for more information

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

    def __init__(self, X, groups=None, n_cond=1, **kwargs):
        # if groups aren't provided or are provided wrong, fix it
        if groups is None:
            groups = [len(X)]
        elif not isinstance(groups, (list, np.ndarray)):
            groups = [groups]
        self.inputs = PLSInputs(X=X, groups=groups, n_cond=n_cond,
                                **kwargs)
        self.rs = check_random_state(self.inputs.seed)

    def gen_covcorr(self, X, Y, groups):
        """
        Should generate cross-covariance array to be used in ``self._svd()``

        Must accept the listed parameters and return one array

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Behavioral matrix, where ``S`` is observations and ``T`` is
            features
        groups : (G,) array_like
            Array with number of subjects in each of ``G`` groups

        Returns
        -------
        crosscov : np.ndarray
            Covariance array for decomposition
        """

        raise NotImplementedError

    def run_pls(self, X, Y):
        """
        Runs PLS analysis

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Behavioral matrix, where ``S`` is observations and ``T`` is
            features
        groups : (G,) array_like
            Array with number of subjects in each of ``G`` groups
        """

        res = PLSResults(inputs=self.inputs)

        # get original singular vectors / values and variance explained
        res.u, res.s, res.v = self.svd(X, Y, seed=self.rs)
        res.s_varexp = compute.crossblock_cov(res.s)

        # compute permutations and get LV significance; store permsamp
        d_perm, ucorrs, vcorrs = self.permutation(X, Y)
        sp, sprob = compute.perm_sig(res.s, d_perm)
        res.perm_result.update(dict(sp=sp, sprob=sprob))

        # get split half reliability results
        if self.inputs.n_split is not None:
            di = np.linalg.inv(res.s)
            ud, vd = res.u @ di, res.v @ di
            orig_ucorr, orig_vcorr = self.split_half(X, Y, ud, vd)
            # get probabilties for ucorr/vcorr
            ucorr_prob = compute.perm_sig(np.diag(orig_ucorr), ucorrs)[-1]
            vcorr_prob = compute.perm_sig(np.diag(orig_vcorr), vcorrs)[-1]
            # get confidence intervals for ucorr/vcorr
            ucorr_ll, ucorr_ul = compute.boot_ci(ucorrs, ci=self.inputs.ci)
            vcorr_ll, vcorr_ul = compute.boot_ci(vcorrs, ci=self.inputs.ci)
            # update results object
            res.perm_splithalf.update(dict(orig_ucorr=orig_ucorr,
                                           orig_vcorr=orig_vcorr,
                                           ucorr_prob=ucorr_prob,
                                           vcorr_prob=vcorr_prob,
                                           ucorr_ll=ucorr_ll,
                                           vcorr_ll=vcorr_ll,
                                           ucorr_ul=ucorr_ul,
                                           vcorr_ul=vcorr_ul))

        return res

    def svd(self, X, Y, dummy=None, seed=None):
        """
        Runs SVD on cross-covariance matrix computed from ``X`` and ``Y``

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Behavioral matrix, where ``S`` is observations and ``T`` is
            features
        seed : {int, RandomState instance, None}, optional
            Seed for pseudo-random number generation. Default: None

        Returns
        -------
        U : (B x L) np.ndarray
            Left singular vectors
        d : (L x L) np.ndarray
            Diagonal array of singular values
        V : (J x L) np.ndarray
            Right singular vectors
        """

        if dummy is None:
            dummy = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        crosscov = self.gen_covcorr(X, Y, groups=dummy)
        n_comp = min(min(dummy.squeeze().shape), min(crosscov.shape))
        U, d, V = randomized_svd(crosscov.T,
                                 n_components=n_comp,
                                 random_state=check_random_state(seed))

        return U, np.diag(d), V.T

    def bootstrap(self, X, Y):
        """
        Bootstraps ``X`` and ``Y`` (w/replacement) and recomputes SVD

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Behavioral matrix, where ``S`` is observations and ``T`` is
            features

        Returns
        -------
        U_boot : (B x L x R) np.ndarray
            Left singular vectors, where ``R`` is the number of bootstraps
        V_boot : (J x L x R) np.ndarray
            Right singular vectors, where ``R`` is the number of bootstraps
        """

        # generate bootstrap resampled indices
        self.bootsamp = self.gen_bootsamp()

        # get original values
        U_orig, d_orig, V_orig = self.svd(X, Y, seed=self.rs)
        U_boot = np.zeros(shape=U_orig.shape + (self.inputs.n_boot,))
        V_boot = np.zeros(shape=V_orig.shape + (self.inputs.n_boot,))

        for i in utils.trange(self.inputs.n_boot, desc='Running bootstraps'):
            inds = self.bootsamp[:, i]
            U, d, V = self.svd(X[inds], Y[inds], seed=self.rs)
            U_boot[:, :, i], rotate = compute.procrustes(U_orig, U, d)
            V_boot[:, :, i] = V @ d @ rotate

        return U_boot, V_boot

    def permutation(self, X, Y):
        """
        Permutes ``X`` and ``Y`` (w/o replacement) and recomputes SVD

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Behavioral matrix, where ``S`` is observations and ``T`` is
            features

        Returns
        -------
        d_perm : (L x P) np.ndarray
            Permuted singular values, where ``L`` is the number of singular
            values and ``P`` is the number of permutations
        ucorrs : (L x P) np.ndarray
            Split-half correlations of left singular values. Only set if
            ``self.inputs.n_split != 0``
        vcorrs : (L x P) np.ndarray
            Split-half correlations of right singular values. Only set if
            ``self.inputs.n_split != 0``
        """

        # generate permuted indices
        self.X_perms, self.Y_perms = self.gen_permsamp()

        # get original values
        U_orig, d_orig, V_orig = self.svd(X, Y, seed=self.rs)

        d_perm = np.zeros(shape=(len(d_orig), self.inputs.n_perm))
        ucorrs = np.zeros(shape=(len(d_orig), self.inputs.n_perm))
        vcorrs = np.zeros(shape=(len(d_orig), self.inputs.n_perm))

        for i in utils.trange(self.inputs.n_perm, desc='Running permutations'):
            Xp, Yp = self.X_perms[:, i], self.Y_perms[:, i]
            outputs = self.single_perm(X[Xp], Y[Yp], V_orig)
            d_perm[:, i] = outputs[0]
            if self.inputs.n_split is not None:
                ucorrs[:, i], vcorrs[:, i] = outputs[1:]

        return d_perm, ucorrs, vcorrs

    def single_perm(self, X, Y, original):
        """
        Permutes ``X`` (w/o replacement) and computes SVD of cross-corr matrix

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Behavioral matrix, where ``S`` is observations and ``T`` is
            features
        original : array_like
            Right singular vectors from non-permuted SVD for use in procrustes
            rotation (if applicable)

        Returns
        -------
        ssd : (L,) np.ndarray
            Sum of squared, permuted singular values
        ucorr : (L,) np.ndarray
            Split-half correlations of left singular values. Only set if
            ``self.inputs.n_split != 0``; otherwise, None
        vcorr : (L,) np.ndarray
            Split-half correlations of right singular values. Only set if
            ``self.inputs.n_split != 0``; otherwise, None
        """

        # perform SVD of permuted array
        U, d, V = self.svd(X, Y, seed=self.rs)

        # optionally get rotated/rescaled singular values (or not)
        if self.inputs.rotate:
            ssd = np.sqrt(np.sum(compute.procrustes(original, V, d)[0]**2,
                          axis=0))
        else:
            ssd = np.diag(d)

        # get ucorr/vcorr if split-half resampling requested
        if self.inputs.n_split is not None:
            di = np.linalg.inv(d)
            ucorr, vcorr = self.split_half(X, Y, U @ di, V @ di)
        else:
            ucorr, vcorr = None, None

        return ssd, ucorr, vcorr

    def split_half(self, X, Y, ud, vd):
        """
        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Behavioral matrix, where ``S`` is observations and ``T`` is
            features
        ud : (B x L) array_like
            Left singular vectors, scaled by singular values
        vd : (J x L) array_like
            Right singular vectors, scaled by singular values

        Returns
        -------
        ucorr : (L,) np.ndarray
            Average correlation of left singular vectors across split-halves
        vcorr : (L,) np.ndarray
            Average correlation of right singular vectors across split-halves
        """

        # generate splits
        splitsamp = self.gen_splits().astype(bool)

        # empty arrays to hold split-half correlations
        ucorr = np.zeros(shape=(ud.shape[-1], self.inputs.n_split))
        vcorr = np.zeros(shape=(vd.shape[-1], self.inputs.n_split))

        for i in range(self.inputs.n_split):
            spl = splitsamp[:, i]

            D1 = self.gen_covcorr(X[spl], Y[spl],
                                  groups=utils.dummy_code(
                                      self.inputs.groups,
                                      self.inputs.n_cond)[spl])
            D2 = self.gen_covcorr(X[~spl], Y[~spl],
                                  groups=utils.dummy_code(
                                      self.inputs.groups,
                                      self.inputs.n_cond)[~spl])

            # project cross-covariance matrices onto original SVD to obtain
            # left & right singular vector
            U1, U2 = D1.T @ vd, D2.T @ vd
            V1, V2 = D1 @ ud, D2 @ ud

            # correlate all the singular vectors between split halves
            ucorr[:, i] = [np.corrcoef(u1, u2)[0, 1] for (u1, u2) in
                           zip(U1.T, U2.T)]
            vcorr[:, i] = [np.corrcoef(v1, v2)[0, 1] for (v1, v2) in
                           zip(V1.T, V2.T)]

        # return average correlations for singular vectors across ``n_split``
        return ucorr.mean(axis=-1), vcorr.mean(axis=-1)

    def gen_permsamp(self):
        """ Generates permutation arrays for ``self._permutation()`` """

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
                inds = np.hstack([utils.permute_cols(i, seed=self.rs) for i
                                  in to_permute])
                # generate permutation of subjects across groups
                perm = self.rs.permutation(subj_inds)
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

        return permsamp, np.ones_like(permsamp, dtype=bool)

    def gen_bootsamp(self):
        """ Generates bootstrap arrays for ``self._bootstrap()`` """

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
                        num_subj = curr_grp.size
                        boot[curr_grp] = np.sort(self.rs.choice(curr_grp,
                                                                size=num_subj,
                                                                replace=True),
                                                 axis=0)
                        # make sure bootstrap has enough unique subjs
                        if np.unique(boot[curr_grp]).size >= min_subj:
                            all_same = False
                # resample subjects (with conditions) and stack groups
                bootinds = np.hstack([f.flatten('F') for f in
                                      np.split(inds[:, boot].T, splitinds)])
                # make sure bootstrap is not a duplicated sequence
                for grp in check_grps:
                    curr_grp = subj_inds[grp]
                    check = bootinds[curr_grp, None] == bootsamp[curr_grp, :i]
                    if check.all(axis=0).any():
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

    def gen_splits(self, test_size=0.5):
        """ Generates split-half arrays for ``self._split_half()`` """

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
                    take = self.rs.choice([np.ceil, np.floor])
                    num_subj = int(take(curr_grp.size * (1 - test_size)))
                    splinds = self.rs.choice(curr_grp,
                                             size=num_subj,
                                             replace=False)
                    split[splinds] = True
                # split subjects (with conditions) and stack groups
                half = np.hstack([f.flatten('F') for f in
                                  np.split(((inds + 1).astype(bool) *
                                            [split[None]]).T,
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
