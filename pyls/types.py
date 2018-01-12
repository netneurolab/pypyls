# -*- coding: utf-8 -*-

import numpy as np
from pyls.base import BasePLS
from pyls import compute, utils


class BehavioralPLS(BasePLS):
    """
    Runs "behavioral" PLS

    Uses singular value decomposition (SVD) to find latent variables (LVs) in
    the cross-covariance matrix of ``brain`` and ``behav``, two subject (N) by
    feature (K) arrays, optionally identifying the differences in these LVs
    between ``groups``. Permutation testing is used to examine statistical
    significance and split-half resampling is used to assess reliability of
    LVs. Bootstrap resampling is used to examine reliability of features (K)
    across LVs.

    Parameters
    ----------
    brain : (N x K) array_like
        Where ``N`` is the number of subjects and ``K`` is the number of
        observations
    behav : (N x J) array_like
        Where ``N`` is the number of subjects and ``J`` is the number of
        observations
    groups : (N,) array_like, optional
        Array with labels separating ``N`` subjects into ``G`` groups. Default:
        None (only one group)
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

    Attributes
    ----------
    U : (K[*G] x L) np.ndarray
        Left singular vectors from decomposition, where ``L`` is the number of
        latent variables identified in the data
    d : (L x L) np.ndarray
        Diagonal array of singular values from decomposition, where ``L`` is
        the number of latent variables identified in the data
    V : (J x L) np.ndarray
        Right singular vectors from decomposition, where ``L`` is the number of
        latent variables identified in the data
    d_pvals : (L,) np.ndarray
        Statistical significance of latent variables as determined by
        permutation testing
    d_varexp : (L,) np.ndarray
        Variance explained by each latent variable
    U_bsr : (K[*G] x L) np.ndarray
        Bootstrap ratios of left singular vectors, as determined by bootstrap
        resampling. Values can be treated as a Z-score, indicating how
        reliably the given feature contributes to the corresponding latent
        variable.
    V_bsr : (J x L) np.ndarray
        Bootstrap ratios of right singular vectors, as determined by bootstrap
        resampling. Values can be treated as a Z-score, indicating how
        reliably the given feature contributes to the corresponding latent
        variable.
    U_corr : (L,) np.ndarray
        Only present if ``n_split`` was set at instantiation. The correlation
        of left singular vectors across split-half resamples in the original
        data.
    V_corr : (L,) np.ndarray
        Only present if ``n_split`` was set at instantiation. The correlation
        of left singular vectors across split-half resamples in the original
        data.
    U_pvals : (L,) np.ndarray
        Only present if ``n_split`` was set at instantiation. Statistical
        significance of the left singular vectors as determined by permutation
        tests across split-half resamples.
    V_pvals : (L,) np.ndarray
        Only present if ``n_split`` was set at instantiation. Statistical
        significance of the right singular vectors as determined by permutation
        tests across split-half resamples.

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

    def __init__(self, brain, behav, groups=None, **kwargs):
        super(BehavioralPLS, self).__init__(**kwargs)
        self.inputs._X, self.inputs._Y = brain, behav
        self.inputs._groups = groups
        self._run_pls(brain, behav, groups=groups)

    def _gen_covcorr(self, X, Y, groups=None):
        """
        Computes cross-covariance matrix from ``X`` and ``Y``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Input array, where ``N`` is the number of subjects and ``J`` is the
            number of variables.
        groups : (N,) array_like, optional
            Array with labels separating ``N`` subjects into ``G`` groups.
            Default: None (only one group)

        Returns
        -------
        cross_cov : (J[*G] x K) np.ndarray
            Cross-covariance matrix
        """

        if X.ndim != Y.ndim:
            raise ValueError('Number of dimensions of ``X`` and ``Y`` must '
                             'match.')
        if X.ndim != 2:
            raise ValueError('``X`` and ``Y`` must each have 2 dimensions.')
        if X.shape[0] != Y.shape[0]:
            raise ValueError('The first dimension of ``X`` and ``Y`` must '
                             'match.')

        if groups is None:
            cross_cov = utils.xcorr(utils.normalize(X),
                                    utils.normalize(Y))
        else:
            cross_cov = [utils.xcorr(utils.normalize(X[groups == grp]),
                                     utils.normalize(Y[groups == grp]))
                         for grp in np.unique(groups)]
            cross_cov = np.row_stack(cross_cov)

        return cross_cov

    def _gen_permsamp(self, X, Y, groups=None):
        """
        Generates permutation arrays to be used in ``self._permutation()``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Input array, where ``N`` is the number of subjects and ``J`` is the
            number of variables.
        groups : (N,) array_like
            Array with labels separating ``N`` subjects into ``G`` groups.
            Default: None (only one group)

        Returns
        -------
        permsamp : (N x P) np.ndarray
        """

        permsamp = np.zeros(shape=(len(X), self.inputs.n_perm), dtype=int)
        subj_inds = np.arange(len(X), dtype=int)

        for i in utils.trange(self.inputs.n_perm, desc='Making permutations'):
            count, duplicated = 0, True
            while duplicated and count < 500:
                # initial permutation attempt
                perm = self._rs.permutation(subj_inds)
                count, duplicated = count + 1, False
                if groups is not None:
                    # iterate through groups and ensure that we aren't just
                    # permuting subjects *within* any of the groups
                    for grp in utils.dummy_code(groups).T.astype(bool):
                        if np.all(np.sort(perm[grp]) == subj_inds[grp]):
                            duplicated = True
                # make sure permutation is not a duplicated sequence
                dupe_seq = perm[:, None] == permsamp[:, :i]
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500:
                print('ERROR: Duplicate permutations used.')
            permsamp[:, i] = perm

        return permsamp

    def _gen_bootsamp(self, X, Y, groups=None):
        """
        Generates bootstrap resample arrays to be used in ``self._bootstrap()``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Input array, where ``N`` is the number of subjects and ``J`` is the
            number of variables.
        groups : (N,) array_like
            Array with labels separating ``N`` subjects into ``G`` groups.
            Default: None (only one group)

        Returns
        -------
        bootsamp : (N x B) np.ndarray
        """

        bootsamp = np.zeros(shape=(len(X), self.inputs.n_boot), dtype=int)
        min_subj = int(np.ceil(Y.sum(axis=0).min() * 0.5))
        subj_inds = np.arange(len(X), dtype=int)

        for i in utils.trange(self.inputs.n_boot, desc='Making bootstraps'):
            count, duplicated = 0, True
            while duplicated and count < 500:
                # empty container to store current bootstrap attempt
                boot = np.zeros(shape=(subj_inds.size, 1), dtype=int)
                count, duplicated = count + 1, False
                if groups is not None:
                    # iterate through and resample from w/i groups
                    for grp in utils.dummy_code(groups).T.astype(bool):
                        curr_grp, all_same = subj_inds[grp], True
                        while all_same:
                            boot[curr_grp, 0] = self._rs.choice(curr_grp,
                                                                size=curr_grp.size,
                                                                replace=True)
                            # make sure bootstrap has enough unique subjs
                            if np.unique(boot[curr_grp]).size >= min_subj:
                                all_same = False
                else:
                    boot[subj_inds, 0] = self._rs.choice(subj_inds,
                                                         size=subj_inds.size,
                                                         replace=True)
                # make sure bootstrap is not a duplicated sequence
                dupe_seq = np.sort(boot) == np.sort(bootsamp[:, :i], axis=0)
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500:
                print('ERROR: Duplicate boostraps used.')
            bootsamp[:, i] = boot.squeeze()

        return bootsamp

    def _gen_splits(self, X, Y, groups=None):
        """
        Generates split half arrays to be using in ``self._split_half()``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Input array, where ``N`` is the number of subjects and ``J`` is the
            number of variables.
        groups : (N,) array_like
            Array with labels separating ``N`` subjects into ``G`` groups.
            Default: None (only one group)

        Returns
        -------
        splitsamp : (N x S) np.ndarray
        """

        splitsamp = np.zeros(shape=(len(X), self.inputs.n_split), dtype=bool)
        subj_inds = np.arange(len(X), dtype=int)

        for i in range(self.inputs.n_split):
            count, duplicated = 0, True
            while duplicated and count < 500:
                # empty containter to store current split half attempt
                split = np.zeros(shape=(subj_inds.size, 1), dtype=bool)
                count, duplicated = count + 1, False
                if groups is not None:
                    # iterate through and split each group separately
                    for grp in utils.dummy_code(groups).T.astype(bool):
                        curr_grp = subj_inds[grp]
                        take = self._rs.choice([np.ceil, np.floor])
                        num_subj = int(take(np.sum(curr_grp.size)/2))
                        inds = self._rs.choice(curr_grp,
                                               size=num_subj,
                                               replace=False)
                        split[inds] = True
                else:
                    inds = self._rs.choice(subj_inds,
                                           size=subj_inds.size // 2,
                                           replace=False)
                    split[inds] = True
                # make sure split half is not a duplicated sequence
                dupe_seq = split == splitsamp[:, :i]
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500:
                print('ERROR: Duplicate split halves used.')
            splitsamp[:, i] = split.squeeze()

        return splitsamp

    def _run_pls(self, X, Y, groups=None):
        """
        Runs PLS analysis

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Input array, where ``N`` is the number of subjects and ``J`` is the
            number of variables.
        groups : (N,) array_like
            Array with labels separating ``N`` subjects into ``G`` groups.
            Default: None (only one group)
        """

        # original singular vectors / values
        self.U, self.d, self.V = self._svd(X, Y, groups=groups,
                                           seed=self._rs)
        # get variance explained by latent variables
        self.d_varexp = compute.crossblock_cov(self.d)

        # compute permutations
        d_perm, ucorrs, vcorrs = self._permutation(X, Y, groups=groups)
        # get LV significance
        self.d_pvals = compute.perm_sig(self.d, d_perm)

        # get split half reliability, if set
        if self.inputs.n_split is not None:
            di = np.linalg.inv(self.d)
            ud, vd = self.U @ di, self.V @ di
            self.U_corr, self.V_corr = self._split_half(X, Y, ud, vd,
                                                        groups=groups)
            self.U_pvals = compute.perm_sig(np.diag(self.U_corr), ucorrs)
            self.V_pvals = compute.perm_sig(np.diag(self.V_corr), vcorrs)

        # compute bootstraps
        U_boot, V_boot = self._bootstrap(X, Y, groups=groups)

        # compute bootstrap ratios
        self.U_bsr = compute.boot_rel(self.U @ self.d, U_boot)
        self.V_bsr = compute.boot_rel(self.V @ self.d, V_boot)


class MeanCenteredPLS(BasePLS):
    """
    Runs "mean-centered" PLS

    Uses singular value decomposition (SVD) to find latent variables (LVs) in
    ``data``, a subject (N) x feature (K) array, that maximize the difference
    between ``groups``. Permutation testing is used to examine statistical
    significance and split-half resampling is used to assess reliability of
    LVs. Bootstrap resampling is used to examine reliability of features (K)
    across LVs.

    Parameters
    ----------
    data : (N x K) array_like
        Original data array where ``N`` is the number of subjects and ``K`` is
        the number of observations
    groups : (N,) array_like
        Array with labels separating ``N`` subjects into ``G`` groups
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

    Attributes
    ----------
    U : (G x L) np.ndarray
        Left singular vectors from decomposition, where ``G`` is the number of
        groups and ``L`` is the number of latent variables identified in the
        data
    d : (L x L) np.ndarray
        Diagonal array of singular values from decomposition, where ``L`` is
        the number of latent variables identified in the data
    V : (J x L) np.ndarray
        Right singular vectors from decomposition, where ``L`` is the number of
        latent variables identified in the data
    d_pvals : (L,) np.ndarray
        Statistical significance of latent variables as determined by
        permutation testing
    d_varexp : (L,) np.ndarray
        Variance explained by each latent variable
    U_bsr : (G x L) np.ndarray
        Bootstrap ratios of left singular vectors, as determined by bootstrap
        resampling. Values can be treated as a Z-score, indicating how
        reliably the given feature contributes to the corresponding latent
        variable.
    V_bsr : (J x L) np.ndarray
        Bootstrap ratios of right singular vectors, as determined by bootstrap
        resampling. Values can be treated as a Z-score, indicating how
        reliably the given feature contributes to the corresponding latent
        variable.
    usc : (N x L) np.ndarray
        "Brainscores" reflecting the extent to which each subject adheres to
        the identified latent variable.
    orig_usc : (G x L) np.ndarray
        Contrast reflecting weighted dissociation of ``groups`` from each other
        for each identified latent variable.
    orig_usc_ll : (G x L) np.ndarray
        Lower bound of confidence interval for ``orig_usc`` as determined by
        bootstrap resampling. CI set by ``ci`` at instantiation.
    orig_usc_ul : (G x L) np.ndarray
        Upper bound of confidence interval for ``orig_usc`` as determined by
        bootstrap resampling. CI set by ``ci`` at instantiation.
    distrib : (G x L x B) np.ndarray
        Distribution of ``orig_usc`` as determined by bootstrap resampling.
        Used to calculated ``orig_usc_ll` and ``orig_usc_ul``.
    U_corr : (L,) np.ndarray
        Only present if ``n_split`` was set at instantiation. The correlation
        of left singular vectors across split-half resamples in the original
        data.
    V_corr : (L,) np.ndarray
        Only present if ``n_split`` was set at instantiation. The correlation
        of left singular vectors across split-half resamples in the original
        data.
    U_pvals : (L,) np.ndarray
        Only present if ``n_split`` was set at instantiation. Statistical
        significance of the left singular vectors as determined by permutation
        tests across split-half resamples.
    V_pvals : (L,) np.ndarray
        Only present if ``n_split`` was set at instantiation. Statistical
        significance of the right singular vectors as determined by permutation
        tests across split-half resamples.

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

    def __init__(self, data, groups, **kwargs):
        super(MeanCenteredPLS, self).__init__(**kwargs)
        # for consistency, assign variables to X and Y
        self.inputs._X, self.inputs._Y = data, utils.dummy_code(groups)
        self.inputs._groups = groups
        # run analysis
        self._run_pls(self.inputs.X, self.inputs.Y)

    def _gen_covcorr(self, X, Y, groups=None):
        """
        Computes mean-centered matrix from ``X`` and ``Y``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Dummy coded input array, where ``N`` is the number of subjects and
            ``J`` corresponds to the number of groups. A value of 1 indicates
            that a subject (row) belongs to a group (column).

        Returns
        -------
        mean_centered : (J x K) np.ndarray
            Mean-centered matrix
        """

        iden = np.ones(shape=(len(Y), 1))
        grp_means = np.linalg.inv(np.diag((iden.T @ Y).flatten())) @ Y.T @ X
        num_group = len(grp_means)
        L = np.ones(shape=(num_group, 1))
        # effectively the same as M - M.mean(axis=0)...
        mean_centered = grp_means - (L @ (((1/num_group) * L.T) @ grp_means))

        return mean_centered

    def _gen_permsamp(self, X, Y, groups=None):
        """
        Generates permutation arrays to be used in ``self._permutation()``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Dummy coded input array, where ``N`` is the number of subjects and
            ``J`` corresponds to the number of groups. A value of 1 indicates
            that a subject (row) belongs to a group (column).
        groups : placeholder

        Returns
        -------
        permsamp : (N x P) np.ndarray
        """

        permsamp = np.zeros(shape=(len(X), self.inputs.n_perm), dtype=int)
        subj_inds = np.arange(len(X), dtype=int)

        for i in utils.trange(self.inputs.n_perm, desc='Making permutations'):
            count, duplicated = 0, True
            while duplicated and count < 500:
                # initial permutation attempt
                perm = self._rs.permutation(subj_inds)
                count, duplicated = count + 1, False
                # iterate through groups and ensure that we aren't just
                # permuting subjects *within* any of the groups
                for grp in Y.T.astype(bool):
                    if np.all(np.sort(perm[grp]) == subj_inds[grp]):
                        duplicated = True
                # make sure permutation is not a duplicated sequence
                dupe_seq = perm[:, None] == permsamp[:, :i]
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500:
                print('ERROR: Duplicate permutations used.')
            permsamp[:, i] = perm

        return permsamp

    def _gen_bootsamp(self, X, Y, groups=None):
        """
        Generates bootstrap resample arrays to be used in ``self._bootstrap()``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Dummy coded input array, where ``N`` is the number of subjects and
            ``J`` corresponds to the number of groups. A value of 1 indicates
            that a subject (row) belongs to a group (column).
        groups : placeholder

        Returns
        -------
        bootsamp : (N x B) np.ndarray
        """

        bootsamp = np.zeros(shape=(len(X), self.inputs.n_boot), dtype=int)
        min_subj = int(np.ceil(Y.sum(axis=0).min() * 0.5))
        subj_inds = np.arange(len(X), dtype=int)

        for i in utils.trange(self.inputs.n_boot, desc='Making bootstraps'):
            count, duplicated = 0, True
            while duplicated and count < 500:
                # empty container to store current bootstrap attempt
                boot = np.zeros(shape=(subj_inds.size, 1), dtype=int)
                count, duplicated = count + 1, False
                # iterate through and resample from w/i groups
                for grp in Y.T.astype(bool):
                    curr_grp, all_same = subj_inds[grp], True
                    while all_same:
                        boot[curr_grp, 0] = self._rs.choice(curr_grp,
                                                            size=curr_grp.size,
                                                            replace=True)
                        # make sure bootstrap has enough unique subjs
                        if np.unique(boot[curr_grp]).size >= min_subj:
                            all_same = False
                # make sure bootstrap is not a duplicated sequence
                dupe_seq = np.sort(boot) == np.sort(bootsamp[:, :i], axis=0)
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500:
                print('ERROR: Duplicate boostraps used.')
            bootsamp[:, i] = boot.squeeze()

        return bootsamp

    def _gen_splits(self, X, Y, groups=None):
        """
        Generates split half arrays to be using in ``self._split_half()``

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Dummy coded input array, where ``N`` is the number of subjects and
            ``J`` corresponds to the number of groups. A value of 1 indicates
            that a subject (row) belongs to a group (column).
        groups : placeholder

        Returns
        -------
        splitsamp : (N x S) np.ndarray
        """

        splitsamp = np.zeros(shape=(len(X), self.inputs.n_split), dtype=bool)
        subj_inds = np.arange(len(X), dtype=int)

        for i in range(self.inputs.n_split):
            count, duplicated = 0, True
            while duplicated and count < 500:
                # empty containter to store current split half attempt
                split = np.zeros(shape=(subj_inds.size, 1), dtype=bool)
                count, duplicated = count + 1, False
                # iterate through and split each group separately
                for grp in Y.T.astype(bool):
                    curr_grp = subj_inds[grp]
                    take = self._rs.choice([np.ceil, np.floor])
                    num_subj = int(take(np.sum(curr_grp.size)/2))
                    inds = self._rs.choice(curr_grp,
                                           size=num_subj,
                                           replace=False)
                    split[inds] = True
                # make sure split half is not a duplicated sequence
                dupe_seq = split == splitsamp[:, :i]
                if dupe_seq.all(axis=0).any():
                    duplicated = True
            if count == 500:
                print('ERROR: Duplicate split halves used.')
            splitsamp[:, i] = split.squeeze()

        return splitsamp

    def _boot_distrib(self, X, Y, V_boot):
        """
        Generates bootstrapped distribution for contrast

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Dummy coded input array, where ``N`` is the number of subjects and
            ``J`` corresponds to the number of groups. A value of 1 indicates
            that a subject (row) belongs to a group (column).
        V_boot : (K x L x B) array_like
            Bootstrapped values of the right singular vectors, where ``L`` is
            the number of latent variables and ``B`` is the number of
            bootstraps

        Returns
        -------
        distrib : (G x L x B) np.ndarray
        """

        distrib = np.zeros(shape=(self.U.shape + (self.inputs.n_boot,)))
        normed_V_boot = utils.normalize(V_boot)

        for i in utils.trange(self.inputs.n_boot, desc='Calculating CI'):
            boot, V = self.bootsamp[:, i], normed_V_boot[:, :, i]
            usc = compute.get_mean_norm(X[boot], Y) @ V
            distrib[:, :, i] = compute.get_group_mean(usc, Y, grand=False)

        return distrib

    def _run_pls(self, X, Y):
        """
        Runs PLS analysis

        Parameters
        ----------
        X : (N x K) array_like
            Input array, where ``N`` is the number of subjects and ``K`` is the
            number of variables.
        Y : (N x J) array_like
            Dummy coded input array, where ``N`` is the number of subjects and
            ``J`` corresponds to the number of groups. A value of 1 indicates
            that a subject (row) belongs to a group (column).
        """

        # original singular vectors / values
        self.U, self.d, self.V = self._svd(X, Y, seed=self._rs)
        # get variance explained by latent variables
        self.d_varexp = compute.crossblock_cov(self.d)

        # compute permutations
        d_perm, ucorrs, vcorrs = self._permutation(X, Y)
        # get LV significance
        self.d_pvals = compute.perm_sig(self.d, d_perm)

        # get split half reliability, if requested
        if self.inputs.n_split is not None:
            di = np.linalg.inv(self.d)
            ud, vd = self.U @ di, self.V @ di
            self.U_corr, self.V_corr = self._split_half(X, Y, ud, vd)
            self.U_pvals = compute.perm_sig(np.diag(self.U_corr), ucorrs)
            self.V_pvals = compute.perm_sig(np.diag(self.V_corr), vcorrs)

        # generate bootstrapped singular vectors
        U_boot, V_boot = self._bootstrap(X, Y)

        # compute bootstrap ratios
        self.U_bsr = compute.boot_rel(self.U @ self.d, U_boot)
        self.V_bsr = compute.boot_rel(self.V @ self.d, V_boot)

        # get normalized brain scores and contrast
        self.usc = compute.get_mean_norm(X, Y) @ self.V
        self.orig_usc = compute.get_group_mean(self.usc, Y, grand=False)

        # generate distribution / confidence intervals for contrast
        self.distrib = self._boot_distrib(X, Y, V_boot)
        self.orig_usc_ll, self.orig_usc_ul = compute.boot_ci(self.distrib,
                                                             ci=self.inputs.ci)
