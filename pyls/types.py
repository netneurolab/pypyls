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
    behavior : (N x J) array_like
        Where ``N`` is the number of subjects and ``J`` is the number of
        observations
    groups : (N,) array_like, optional
        Array with labels separating ``N`` subjects into ``G`` groups. Default:
        None (only one group)
    n_cond : int, optional
        Number of conditions. Default: 1
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

    def __init__(self, brain, behavior, **kwargs):
        X, Y = np.asarray(brain), np.asarray(behavior)
        super().__init__(X=X, Y=Y, **kwargs)
        self._run_pls(self.inputs.X, self.inputs.Y)

    def _gen_covcorr(self, X, Y, groups):
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

        if groups.shape[-1] == 1:
            cross_cov = utils.xcorr(utils.normalize(X),
                                    utils.normalize(Y))
        else:
            cross_cov = [utils.xcorr(utils.normalize(X[grp]),
                                     utils.normalize(Y[grp]))
                         for grp in groups.T.astype(bool)]
            cross_cov = np.row_stack(cross_cov)

        return cross_cov

    def _run_pls(self, X, Y):
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
        self.U, self.d, self.V = self._svd(X, Y, seed=self._rs)
        # get variance explained by latent variables
        self.d_varexp = compute.crossblock_cov(self.d)

        # compute permutations
        d_perm, ucorrs, vcorrs = self._permutation(X, Y)
        # get LV significance
        self.d_pvals = compute.perm_sig(self.d, d_perm)

        # get split half reliability, if set
        if self.inputs.n_split is not None:
            di = np.linalg.inv(self.d)
            ud, vd = self.U @ di, self.V @ di
            self.U_corr, self.V_corr = self._split_half(X, Y, ud, vd)
            self.U_pvals = compute.perm_sig(np.diag(self.U_corr), ucorrs)
            self.V_pvals = compute.perm_sig(np.diag(self.V_corr), vcorrs)

        # compute bootstraps
        U_boot, V_boot = self._bootstrap(X, Y)

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
    groups : (G,) list
        List with number of subjects in each of ``G`` groups
    n_cond : int, optional
        Number of conditions. Default: 1
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
        super().__init__(X=np.array(data), groups=groups, **kwargs)
        self.inputs.Y = utils.dummy_code(self.inputs.groups,
                                         self.inputs.n_cond)
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
            Dummy coded input array, where ``C`` is the number of subjects x
            the number of conditions, and ``J`` corresponds to the number of
            groups x conditions. A value of 1 indicates that a subject
            condition (row) belongs to a specific condition group (column).

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
