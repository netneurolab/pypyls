# -*- coding: utf-8 -*-

from textwrap import dedent
import warnings
import numpy as np
from sklearn.metrics import r2_score
from pyls.base import BasePLS, _pls_input_docs
from pyls import compute, utils


class BehavioralPLS(BasePLS):
    def __init__(self, X, Y, groups=None, n_cond=1, mean_centering=0,
                 n_perm=5000, n_boot=5000, n_split=100, test_size=0.25,
                 rotate=True, ci=95, seed=None, **kwargs):

        super().__init__(X=np.asarray(X), Y=np.asarray(Y), groups=groups,
                         n_cond=n_cond, mean_centering=mean_centering,
                         n_perm=n_perm, n_boot=n_boot, n_split=n_split,
                         test_size=test_size, rotate=rotate, ci=ci, seed=seed)
        self.results = self.run_pls(self.inputs.X, self.inputs.Y)

    def gen_covcorr(self, X, Y, groups, **kwargs):
        """
        Computes cross-covariance matrix from `X` and `Y`

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        groups : (S, J) array_like
            Dummy coded input array, where `S` is observations and `J`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.

        Returns
        -------
        crosscov : (J*T, B) np.ndarray
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
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        U_boot : (K, L, B) array_like
            Bootstrapped values of the right singular vectors, where `L` is the
            number of latent variables and `B` is the number of bootstraps
        groups : (S, J) array_like
            Dummy coded input array, where `S` is observations and `J`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.

        Returns
        -------
        distrib : (G, L, B) np.ndarray
        """

        distrib = np.zeros(shape=(groups.shape[-1] * Y.shape[-1],
                                  U_boot.shape[1],
                                  self.inputs.n_boot,))

        for i in utils.trange(self.inputs.n_boot, desc='Calculating CI'):
            boot = self.bootsamp[:, i]
            tusc = X[boot] @ utils.normalize(U_boot[:, :, i])
            distrib[:, :, i] = self.gen_covcorr(tusc, Y[boot], groups)

        return distrib

    def crossval(self, X, Y):
        """
        Performs cross-validation of SVD of `X` and `Y`

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features

        Returns
        -------
        r_scores : (C,) np.ndarray
            R (Pearon correlation) scores across train-test splits
        r2_scores : (C,) np.ndarray
            R^2 (coefficient of determination) scores across train-test splits
        """

        # use gen_splits to handle grouping/condition vars in train/test split
        splits = self.gen_splits(test_size=self.inputs.test_size)
        dummy = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        r_scores = np.zeros((Y.shape[-1], self.inputs.n_split))
        r2_scores = np.zeros((Y.shape[-1], self.inputs.n_split))

        for i in utils.trange(self.inputs.n_split, desc='Running cross-val'):
            # subset appropriately into train/test sets
            split = splits[:, i]
            X_train, Y_train, dummy_train = X[split], Y[split], dummy[split]
            X_test, Y_test, dummy_test = X[~split], Y[~split], dummy[~split]
            # perform initial decomposition on train set
            U, d, V = self.svd(X_train, Y_train,
                               dummy=dummy_train,
                               seed=self.rs)
            # rescale test set prediction, handling grouping/condition vars
            Y_pred = np.row_stack([compute.rescale_test(X_train[tr_grp],
                                                        X_test[te_grp],
                                                        Y_train[tr_grp],
                                                        U, V_spl)
                                   for V_spl, tr_grp, te_grp in
                                   zip(np.split(V, dummy.shape[-1]),
                                       dummy_train.T.astype(bool),
                                       dummy_test.T.astype(bool))])
            # calculate r & r-squared from comp of rescaled test & true values
            r_scores[:, i] = [np.corrcoef(Y_test[:, i], Y_pred[:, i])[0, 1]
                              for i in range(Y_test.shape[-1])]
            r2_scores[:, i] = r2_score(Y_test, Y_pred,
                                       multioutput='raw_values')

        return r_scores, r2_scores

    def run_pls(self, X, Y):
        """
        Runs PLS analysis

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        """

        res = super().run_pls(X, Y)
        res.perm_result.permsamp = self.Y_perms
        res.usc = X @ res.u
        # mechanism for splitting outputs along group / condition indices
        grps = np.repeat(res.inputs.groups, res.inputs.n_cond)
        res.vsc = np.vstack([y @ v for (y, v) in
                             zip(np.split(Y, np.cumsum(grps)[:-1]),
                                 np.split(res.v, len(grps)))])

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

        # compute cross-validated coefficient of determination
        if self.inputs.n_split is not None and self.inputs.test_size > 0:
            r, r2 = self.crossval(X, Y)
            res.cross_val.update(dict(pearson_r=r, r_squared=r2))

        return res


BehavioralPLS.__doc__ = dedent("""\
    Performs behavioral PLS on `X` and `Y`.

    Behavioral PLS is a multivariate statistical approach that relates two sets
    of variables together. Traditionally, one of these arrays
    represents a set of brain features (e.g., functional connectivity
    estimates) and the other represents a set of behavioral variables; however,
    these arrays can be any two sets of features belonging to a common group of
    samples.

    Using a singular value decomposition, behavioral PLS attempts to find
    linear combinations of features from the provided arrays that maximally
    covary with each other. The decomposition is performed on the cross-
    covariance matrix `R`, where :math:`R = Y' \\times X`, which represents the
    covariation of all the input features across samples.

    {decomposition_narrative}

    Parameters
    ----------
    X : (S, B) array_like
        Input data matrix, where `S` is samples and `B` is features
    Y : (S, T) array_like
        Input data matrix, where `S` is samples and `T` is features
    {groups}
    {conditions}
    {stat_test}
    {rotate}
    {ci}
    {seed}

    Attributes
    ----------
    results : :obj:`pyls.PLSResults`
        Dictionary-like object containing results

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
    .. [5] Misic, B., Betzel, R. F., de Reus, M. A., van den Heuvel, M.P.,
       Berman, M. G., McIntosh, A. R., & Sporns, O. (2016). Network level
       structure-function relationships in human neocortex. Cerebral Cortex,
       26, 3285-96.
    """).format(**_pls_input_docs)


class MeanCenteredPLS(BasePLS):
    def __init__(self, X, groups=None, n_cond=1, mean_centering=0, n_perm=5000,
                 n_boot=5000, n_split=100, test_size=0.25, rotate=True, ci=95,
                 seed=None, **kwargs):
        if groups is None:
            groups = [len(X)]
        # check inputs for validity
        if n_cond == 1 and len(groups) == 1:
            raise ValueError('Cannot perform PLS with only one group and one '
                             'condition. Please confirm inputs are correct.')
        if n_cond == 1 and mean_centering == 0:
            warnings.warn('Cannot set mean_centering to 0 when there is only'
                          'one condition. Resetting mean_centering to 1.')
            mean_centering = 1
        elif len(groups) == 1 and mean_centering == 1:
            warnings.warn('Cannot set mean_centering to 1 when there is only '
                          'one group. Resetting mean_centering to 0.')
            mean_centering = 0

        # instantiate base class, generate dummy array, and run PLS analysis
        super().__init__(X=np.asarray(X), groups=groups, n_cond=n_cond,
                         mean_centering=mean_centering, n_perm=n_perm,
                         n_boot=n_boot, n_split=n_split, test_size=test_size,
                         rotate=rotate, ci=ci, seed=seed)
        self.inputs.Y = utils.dummy_code(self.inputs.groups,
                                         self.inputs.n_cond)
        self.results = self.run_pls(self.inputs.X, self.inputs.Y)

    def gen_covcorr(self, X, Y, **kwargs):
        """
        Computes mean-centered matrix from `X` and `Y`

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Dummy coded input array, where `S` is observations and `T`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.

        Returns
        -------
        mean_centered : (T, B) np.ndarray
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
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Dummy coded input array, where `S` is observations and `T`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.
        U_boot : (B, L, R) array_like
            Bootstrapped values of the right singular vectors, where `B` is the
            same as in `X`, `L` is the number of latent variables and `R` is
            the number of bootstraps

        Returns
        -------
        distrib : (T, L, R) np.ndarray
        """

        distrib = np.zeros(shape=(Y.shape[-1], U_boot.shape[1],
                                  self.inputs.n_boot,))

        for i in range(self.inputs.n_boot):
            boot, U = self.bootsamp[:, i], U_boot[:, :, i]
            usc = compute.get_mean_center(X[boot], Y,
                                          self.inputs.n_cond,
                                          self.inputs.mean_centering,
                                          means=False) @ utils.normalize(U)
            distrib[:, :, i] = np.row_stack([usc[grp].mean(axis=0) for grp
                                             in Y.T.astype(bool)])

        return distrib

    def run_pls(self, X, Y):
        """
        Runs PLS analysis

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like, optional
            Dummy coded input array, where `S` is observations and `T`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.
        """

        res = super().run_pls(X, Y)
        res.perm_result.permsamp = self.X_perms
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


MeanCenteredPLS.__doc__ = dedent("""\
    Performs mean-centered PLS on `X`, sorted into `groups` and `conditions`.

    Mean-centered PLS is a multivariate statistical approach that attempts to
    find sets of variables in a matrix which maximally discriminate between
    groups or conditions. While it carries the name PLS, it is perhaps more
    related to principal components analysis (PCA) than it is to
    :obj:`pyls.BehavioralPLS`. The main differentiation from PCA is that
    `mean_centering` can be set to "boost" or highlight potential differences
    between groups or condition variables.

    {decomposition_narrative}

    Parameters
    ----------
    X : (S, B) array_like
        Input data matrix, where `S` is observations and `B` is features
    {groups}
    {conditions}
    {mean_centering}
    {stat_test}
    {rotate}
    {ci}
    {seed}

    Attributes
    ----------
    results : :obj:`pyls.PLSResults`
        Dictionary-like object containing results

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
    """).format(**_pls_input_docs)
