# -*- coding: utf-8 -*-

import numpy as np
from pyls.base import BasePLS
from pyls import compute, utils


class BehavioralPLS(BasePLS):
    """
    Runs "behavioral" PLS

    Uses singular value decomposition (SVD) to find latent variables (LVs) in
    the cross-covariance matrix of ``X`` and ``Y``, two subject (N) by
    feature (K) arrays, optionally identifying the differences in these LVs
    between ``groups``. Permutation testing is used to examine statistical
    significance and split-half resampling is used to assess reliability of
    LVs. Bootstrap resampling is used to examine reliability of features (K)
    across LVs.

    Parameters
    ----------
    X : (N x K) array_like
        Where ``N`` is the number of subjects and ``K`` is the number of
        observations
    Y : (N x J) array_like
        Where ``N`` is the number of subjects and ``J`` is the number of
        observations

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

    def __init__(self, X, Y, **kwargs):
        super().__init__(X=np.asarray(X), Y=np.asarray(Y), **kwargs)
        self.results = self.run_pls(self.inputs.X, self.inputs.Y)

    def gen_covcorr(self, X, Y, groups):
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
            cross_cov = utils.xcorr(X, Y)
        else:
            cross_cov = [utils.xcorr(X[grp], Y[grp], norm=False)
                         for grp in groups.T.astype(bool)]
            cross_cov = np.row_stack(cross_cov)

        return cross_cov

    def boot_distrib(self, X, Y, U_boot, groups):
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
        U_boot : (K x L x B) array_like
            Bootstrapped values of the right singular vectors, where ``L`` is
            the number of latent variables and ``B`` is the number of
            bootstraps

        Returns
        -------
        distrib : (G x L x B) np.ndarray
        """

        distrib = np.zeros(shape=(groups.shape[-1] * Y.shape[-1],
                                  U_boot.shape[1],
                                  self.inputs.n_boot,))

        for i in utils.trange(self.inputs.n_boot, desc='Calculating CI'):
            boot = self.bootsamp[:, i]
            tusc, yboot = X[boot] @ utils.normalize(U_boot[:, :, i]), Y[boot]
            distrib[:, :, i] = self.gen_covcorr(tusc, yboot, groups)

        return distrib

    def run_pls(self, X, Y):
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

        res = super().run_pls(X, Y)
        res.usc = X @ res.u
        res.vsc = np.vstack([y @ v for (y, v) in
                             zip(np.split(Y, len(res.inputs.groups)),
                                 np.split(res.v, len(res.inputs.groups)))])

        # compute bootstraps and BSRs; store bootsamp
        U_boot, V_boot = self.bootstrap(X, Y)
        compare_u, u_se = compute.boot_rel(res.u @ res.s, U_boot)
        res.boot_result.compare_u, res.boot_result.u_se = compare_u, u_se
        res.boot_result.bootsamp = self.bootsamp

        # get lvcorrs
        groups = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        lvcorrs = self.gen_covcorr(res.usc, Y, groups)
        res.lvcorrs, res.boot_result.orig_corr = lvcorrs, lvcorrs

        # generate distribution / confidence intervals for lvcorrs
        res.boot_result.distrib = self.boot_distrib(X, Y, U_boot, groups)
        llcorr, ulcorr = compute.boot_ci(res.boot_result.distrib,
                                         ci=self.inputs.ci)
        res.boot_result.llcorr, res.boot_result.ulcorr = llcorr, ulcorr

        return res


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
    X : (N x K) array_like
        Original data array where ``N`` is the number of subjects and ``K`` is
        the number of observations
    groups : (G,) list
        List with number of subjects in each of ``G`` groups

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

    def __init__(self, X, groups, **kwargs):
        super().__init__(X=np.asarray(X), groups=groups, **kwargs)
        self.inputs.Y = utils.dummy_code(self.inputs.groups,
                                         self.inputs.n_cond)
        self.results = self.run_pls(self.inputs.X, self.inputs.Y)

    def gen_covcorr(self, X, Y, groups=None):
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

    def boot_distrib(self, X, Y, U_boot):
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
        U_boot : (K x L x B) array_like
            Bootstrapped values of the right singular vectors, where ``L`` is
            the number of latent variables and ``B`` is the number of
            bootstraps

        Returns
        -------
        distrib : (G x L x B) np.ndarray
        """

        distrib = np.zeros(shape=(Y.shape[-1], U_boot.shape[1],
                                  self.inputs.n_boot,))
        normed_U_boot = utils.normalize(U_boot)

        for i in utils.trange(self.inputs.n_boot, desc='Calculating CI'):
            boot, U = self.bootsamp[:, i], normed_U_boot[:, :, i]
            usc = compute.get_mean_norm(X[boot], Y) @ U
            distrib[:, :, i] = compute.get_group_mean(usc, Y, grand=False)

        return distrib

    def run_pls(self, X, Y):
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
        res = super().run_pls(X, Y)
        res.usc, res.vsc = X @ res.u, Y @ res.v

        # compute bootstraps and BSRs; store bootsamp
        U_boot, V_boot = self.bootstrap(X, Y)
        compare_u, u_se = compute.boot_rel(res.u @ res.s, U_boot)
        res.boot_result.compare_u, res.boot_result.u_se = compare_u, u_se
        res.boot_result.bootsamp = self.bootsamp

        # get normalized brain scores and contrast
        res.boot_result.usc2 = compute.get_mean_norm(X, Y) @ res.u
        res.boot_result.orig_usc = compute.get_group_mean(res.boot_result.usc2,
                                                          Y, grand=False)

        # generate distribution / confidence intervals for contrast
        res.boot_result.distrib = self.boot_distrib(X, Y, U_boot)
        llusc, ulusc = compute.boot_ci(res.boot_result.distrib,
                                       ci=self.inputs.ci)
        res.boot_result.llusc, res.boot_result.ulusc = llusc, ulusc

        return res
