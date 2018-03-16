# -*- coding: utf-8 -*-

import warnings
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
    X : (S x B) array_like
        Input data matrix, where ``S`` is observations and ``B`` is features
    Y : (S x T) array_like
        Behavioral matrix, where ``S`` is observations and ``T`` is features
    **kwargs : dict, optional
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

    def __init__(self, X, Y, **kwargs):
        super().__init__(X=np.asarray(X), Y=np.asarray(Y), **kwargs)
        self.results = self.run_pls(self.inputs.X, self.inputs.Y)

    def gen_covcorr(self, X, Y, groups, **kwargs):
        """
        Computes cross-covariance matrix from ``X`` and ``Y``

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Behavioral matrix, where ``S`` is observations and ``T`` is
            features
        groups : (S x J) array_like
            Dummy coded input array, where ``S`` is observations and ``J``
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.

        Returns
        -------
        crosscov : (J*T x B) np.ndarray
            Cross-covariance matrix
        """

        crosscov = np.row_stack([utils.xcorr(X[grp], Y[grp], norm=False)
                                 for grp in groups.T.astype(bool)])

        return crosscov

    def gen_permsamp(self):
        """ Need to flip permutation (i.e., permute Y, not X) """

        Y_perms, X_perms = super().gen_permsamp()

        return X_perms, Y_perms

    def boot_distrib(self, X, Y, U_boot, groups):
        """
        Generates bootstrapped distribution for contrast

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Behavioral matrix, where ``S`` is observations and ``T`` is
            features
        U_boot : (K x L x B) array_like
            Bootstrapped values of the right singular vectors, where ``L`` is
            the number of latent variables and ``B`` is the number of
            bootstraps
        groups : (S x J) array_like
            Dummy coded input array, where ``S`` is observations and ``J``
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.

        Returns
        -------
        distrib : (G x L x B) np.ndarray
        """

        distrib = np.zeros(shape=(groups.shape[-1] * Y.shape[-1],
                                  U_boot.shape[1],
                                  self.inputs.n_boot,))

        for i in utils.trange(self.inputs.n_boot, desc='Calculating CI'):
            boot = self.bootsamp[:, i]
            tusc = X[boot] @ utils.normalize(U_boot[:, :, i])
            distrib[:, :, i] = self.gen_covcorr(tusc, Y[boot], groups)

        return distrib

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
        """

        res = super().run_pls(X, Y)
        res.usc = X @ res.u
        res.vsc = np.vstack([y @ v for (y, v) in
                             zip(np.split(Y, len(res.inputs.groups)),
                                 np.split(res.v, len(res.inputs.groups)))])

        # compute bootstraps and BSRs
        U_boot, V_boot = self.bootstrap(X, Y)
        compare_u, u_se = compute.boot_rel(res.u @ res.s, U_boot)

        # get lvcorrs
        groups = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        res.lvcorrs = self.gen_covcorr(res.usc, Y, groups)

        # generate distribution / confidence intervals for lvcorrs
        distrib = self.boot_distrib(X, Y, U_boot, groups)
        llcorr, ulcorr = compute.boot_ci(distrib, ci=self.inputs.ci)

        # update results.boot_result dictionary
        res.boot_result.update(dict(compare_u=compare_u, u_se=u_se,
                                    bootsamp=self.bootsamp,
                                    orig_corr=res.lvcorrs, distrib=distrib,
                                    llcorr=llcorr, ulcorr=ulcorr))

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
    X : (S x B) array_like
        Input data matrix, where ``S`` is observations and ``B`` is features
    groups : (G,) list
        List with number of subjects in each of ``G`` groups
    mean_centering : int, optional
        Mean centering type. Must be in [0, 1, 2]. Default: 0
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

    def __init__(self, X, groups, mean_centering=0, **kwargs):
        if kwargs.get('n_cond', 1) == 1:
            warnings.warn('Cannot set mean_centering to 0 when n_cond = 1.'
                          'Resetting mean_centering to 1.')
            mean_centering = 1
        super().__init__(X=np.asarray(X),
                         groups=groups,
                         mean_centering=mean_centering,
                         **kwargs)
        self.inputs.Y = utils.dummy_code(self.inputs.groups,
                                         self.inputs.n_cond)
        self.results = self.run_pls(self.inputs.X, self.inputs.Y)

    def gen_covcorr(self, X, Y, **kwargs):
        """
        Computes mean-centered matrix from ``X`` and ``Y``

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Dummy coded input array, where ``S`` is observations and ``T``
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.

        Returns
        -------
        mean_centered : (T x B) np.ndarray
            Mean-centered matrix
        """

        mean_centered = compute.get_mean_center(X, Y,
                                                self.inputs.n_cond,
                                                self.inputs.mean_centering,
                                                means=True)
        return mean_centered

    def boot_distrib(self, X, Y, U_boot):
        """
        Generates bootstrapped distribution for contrast

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like
            Dummy coded input array, where ``S`` is observations and ``T``
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.
        U_boot : (B x L x R) array_like
            Bootstrapped values of the right singular vectors, where ``B`` is
            the same as in ``X``, `L`` is the number of latent variables and
            ``R`` is the number of bootstraps

        Returns
        -------
        distrib : (T x L x R) np.ndarray
        """

        distrib = np.zeros(shape=(Y.shape[-1], U_boot.shape[1],
                                  self.inputs.n_boot,))
        normed_U_boot = utils.normalize(U_boot)

        for i in utils.trange(self.inputs.n_boot, desc='Calculating CI'):
            boot, U = self.bootsamp[:, i], normed_U_boot[:, :, i]
            usc = compute.get_mean_center(X[boot], Y,
                                          self.inputs.n_cond,
                                          self.inputs.mean_centering,
                                          means=False) @ U
            distrib[:, :, i] = np.row_stack([usc[grp].mean(axis=0) for grp
                                             in Y.T.astype(bool)])

        return distrib

    def run_pls(self, X, Y):
        """
        Runs PLS analysis

        Parameters
        ----------
        X : (S x B) array_like
            Input data matrix, where ``S`` is observations and ``B`` is
            features
        Y : (S x T) array_like, optional
            Dummy coded input array, where ``S`` is observations and ``T``
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.
        """
        res = super().run_pls(X, Y)
        res.usc, res.vsc = X @ res.u, Y @ res.v

        # compute bootstraps and BSRs
        U_boot, V_boot = self.bootstrap(X, Y)
        compare_u, u_se = compute.boot_rel(res.u @ res.s, U_boot)

        # get normalized brain scores and contrast
        usc2 = compute.get_mean_center(X, Y,
                                       self.inputs.n_cond,
                                       self.inputs.mean_centering,
                                       means=False) @ res.u
        orig_usc = np.row_stack([usc2[grp].mean(axis=0) for grp
                                 in Y.T.astype(bool)])

        # generate distribution / confidence intervals for contrast
        distrib = self.boot_distrib(X, Y, U_boot)
        llusc, ulusc = compute.boot_ci(distrib,
                                       ci=self.inputs.ci)

        # update results.boot_result dictionary
        res.boot_result.update(dict(compare_u=compare_u, u_se=u_se,
                                    bootsamp=self.bootsamp,
                                    orig_usc=orig_usc, distrib=distrib,
                                    usc2=usc2, llusc=llusc, ulusc=ulusc))

        return res
