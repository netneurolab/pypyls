# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import r2_score
from ..base import BasePLS, gen_splits
from ..structures import _pls_input_docs
from .. import compute, utils


class BehavioralPLS(BasePLS):
    def __init__(self, X, Y, *, groups=None, n_cond=1, n_perm=5000,
                 n_boot=5000, n_split=100, test_size=0.25, covariance=False,
                 rotate=True, ci=95, seed=None, verbose=True, **kwargs):

        super().__init__(X=np.asarray(X), Y=np.asarray(Y), groups=groups,
                         n_cond=n_cond, n_perm=n_perm, n_boot=n_boot,
                         n_split=n_split, test_size=test_size,
                         covariance=covariance, rotate=rotate, ci=ci,
                         seed=seed, verbose=verbose, **kwargs)
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

        crosscov = []
        for grp in groups.T.astype(bool):
            crosscov.append(compute.xcorr(X[grp], Y[grp],
                                          norm=False,
                                          covariance=self.inputs.covariance))

        return np.row_stack(crosscov)

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
            tusc = X[boot] @ compute.normalize(U_boot[:, :, i])
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
        splits = gen_splits(self.inputs.groups,
                            self.inputs.n_cond,
                            self.inputs.n_split,
                            seed=self.rs,
                            test_size=self.inputs.test_size)
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
        res.brainscores = X @ res.u
        # mechanism for splitting outputs along group / condition indices
        grps = np.repeat(res.inputs.groups, res.inputs.n_cond)
        res.behavscores = np.vstack([y @ v for (y, v) in
                                     zip(np.split(Y, np.cumsum(grps)[:-1]),
                                         np.split(res.v, len(grps)))])

        # get lvcorrs
        groups = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        res.behavcorr = self.gen_covcorr(res.brainscores, Y, groups)

        # compute bootstraps and BSRs
        if self.inputs.n_boot > 0:
            U_boot, V_boot = self.bootstrap(X, Y)
            compare_u, u_se = compute.boot_rel(res.u @ res.s, U_boot)

            # generate distribution / confidence intervals for lvcorrs
            distrib = self.boot_distrib(X, Y, U_boot, groups)
            llcorr, ulcorr = compute.boot_ci(distrib, ci=self.inputs.ci)

            # update results.boot_result dictionary
            res.bootres.update(dict(bootstrapratios=compare_u,
                                    uboot_se=u_se,
                                    bootsamples=self.bootsamp,
                                    behavcorr=res.behavcorr,
                                    behavcorr_boot=distrib,
                                    behavcorr_lolim=llcorr,
                                    behavcorr_uplim=ulcorr))

        # compute cross-validated prediction-based metrics
        if self.inputs.n_split is not None and self.inputs.test_size > 0:
            r, r2 = self.crossval(X, Y)
            res.cvres.update(dict(pearson_r=r, r_squared=r2))

        # get rid of the stupid diagonal matrix
        res.s = np.diag(res.s)

        return res


# let's make it a function
def behavioral_pls(X, Y, *, groups=None, n_cond=1, n_perm=5000, n_boot=5000,
                   n_split=100, test_size=0.25, covariance=False, rotate=True,
                   ci=95, seed=None, verbose=True, **kwargs):
    pls = BehavioralPLS(X=X, Y=Y, groups=groups, n_cond=n_cond,
                        n_perm=n_perm, n_boot=n_boot, n_split=n_split,
                        test_size=test_size, covariance=covariance,
                        rotate=rotate, ci=ci, seed=seed, verbose=verbose,
                        **kwargs)
    return pls.results


behavioral_pls.__doc__ = r"""\
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
covariance matrix :math:`R`, where :math:`R = Y^{{T}} \times X`, which
represents the covariation of all the input features across samples.

Parameters
----------
{input_matrix}
Y : (S, T) array_like
    Input data matrix, where `S` is samples and `T` is features
{groups}
{conditions}
{stat_test}
{covariance}
{rotate}
{ci}
{seed}
{verbose}

Returns
----------
{pls_results}

Notes
-----
{decomposition_narrative}

References
----------

{references}

Misic, B., Betzel, R. F., de Reus, M. A., van den Heuvel, M.P.,
Berman, M. G., McIntosh, A. R., & Sporns, O. (2016). Network level
structure-function relationships in human neocortex. Cerebral Cortex,
26, 3285-96.
""".format(**_pls_input_docs)
