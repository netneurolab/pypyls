# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import r2_score
from ..base import BasePLS, gen_splits
from ..structures import _pls_input_docs
from .. import compute, utils


class BehavioralPLS(BasePLS):
    def __init__(self, X, Y, *, groups=None, n_cond=1, n_perm=5000,
                 n_boot=5000, n_split=100, test_size=0.25, test_split=100,
                 covariance=False, rotate=True, ci=95, seed=None, verbose=True,
                 n_proc=None, **kwargs):

        # check that inputs are valid
        if len(X) != len(Y):
            raise ValueError('Provided `X` and `Y` matrices must have the '
                             'same number of samples. Provided matrices '
                             'differed: X: {}, Y: {}'.format(len(X), len(Y)))

        super().__init__(X=np.asarray(X), Y=np.asarray(Y), groups=groups,
                         n_cond=n_cond, n_perm=n_perm, n_boot=n_boot,
                         n_split=n_split, test_size=test_size,
                         test_split=test_split, covariance=covariance,
                         rotate=rotate, ci=ci, seed=seed, verbose=verbose,
                         n_proc=n_proc, **kwargs)

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

    def make_permutation(self, X, Y, perminds):
        """
        Permutes `Y` according to `perminds`, leaving `X` un-permuted

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        perminds : (S,) array_like
            Array by which to permute `Y`

        Returns
        -------
        Xp : (S, B) array_like
            Identical to `X`
        Yp : (S, T) array_like
            `Y`, permuted according to `perminds`
        """

        return X, Y[perminds]

    def boot_distrib(self, X, Y, U_boot, groups, verbose=True):
        """
        Generates bootstrapped distribution for behavioral correlations

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        U_boot : (K, L, B) array_like
            Bootstrapped values of the left singular vectors, where `L` is the
            number of latent variables and `B` is the number of bootstraps
        groups : (S, J) array_like
            Dummy coded input array, where `S` is observations and `J`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.
        verbose : bool, optional
            Whether to print status updates as CI is calculated. Default: True

        Returns
        -------
        distrib : (G, L, B) np.ndarray
        """

        parallel, func = utils.get_par_func(self.inputs.n_proc,
                                            self.__class__._single_distrib)
        generator = utils.trange(self.inputs.n_boot, verbose=verbose,
                                 desc='Calculating CI')
        out = parallel(func(self, X=X, Y=Y, groups=groups,
                            inds=self.bootsamp[:, i],
                            original=U_boot[..., i]) for i in generator)

        return np.stack(out, axis=-1)

    def _single_distrib(self, X, Y, groups, inds, original):
        """
        Finds behavioral correlations for single bootstrap resample

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
        inds : (S,) array_like
            Bootstrap resampling array
        boot : (B, L) array_like
            Left singular vectors from bootstrap

        Returns
        -------
        distrib : (T, L)
            Behavioral correlations for single bootstrap resample
        """

        tusc = X[inds] @ compute.normalize(original)

        return self.gen_covcorr(tusc, Y[inds], groups)

    def crossval(self, X, Y, seed=None, verbose=True):
        """
        Performs cross-validation of SVD of `X` and `Y`

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None
        verbose : bool, optional
            Whether to print status updates as CV is calculated. Default: True

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
                            self.inputs.test_split,
                            seed=seed,
                            test_size=self.inputs.test_size)
        dummy = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)

        parallel, func = utils.get_par_func(self.inputs.n_proc,
                                            self.__class__._single_crossval)
        generator = utils.trange(self.inputs.test_split, verbose=verbose,
                                 desc='Running cross-validation')
        out = parallel(func(self, X=X, Y=Y, dummy=dummy,
                            inds=splits[:, i], seed=i) for i in generator)
        r_scores, r2_scores = [np.stack(o, axis=-1) for o in zip(*out)]

        return r_scores, r2_scores

    def _single_crossval(self, X, Y, dummy, inds, seed=None):
        """
        Generates single cross-validated r and r^2 score

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None
        """

        X_train, Y_train, dummy_train = X[inds], Y[inds], dummy[inds]
        X_test, Y_test, dummy_test = X[~inds], Y[~inds], dummy[~inds]
        # perform initial decomposition on train set
        U, d, V = self.svd(X_train, Y_train, dummy=dummy_train, seed=seed)

        # rescale the test set based on the training set
        Y_pred = []
        for n, V_spl in enumerate(np.split(V, dummy.shape[-1])):
            tr_grp = dummy_train[:, n].astype(bool)
            te_grp = dummy_test[:, n].astype(bool)
            rescaled = compute.rescale_test(X_train[tr_grp], X_test[te_grp],
                                            Y_train[tr_grp], U, V_spl)
            Y_pred.append(rescaled)
        Y_pred = np.row_stack(Y_pred)

        # calculate r & r-squared from comp of rescaled test & true values
        r_scores = compute.efficient_corr(Y_test, Y_pred)
        r2_scores = r2_score(Y_test, Y_pred, multioutput='raw_values')

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
            U_boot, V_boot = self.bootstrap(X, Y, seed=self.rs)
            compare_u, u_se = compute.boot_rel(res.u @ res.s, U_boot)

            # generate distribution / confidence intervals for lvcorrs
            distrib = self.boot_distrib(X, Y, U_boot, groups,
                                        verbose=self.inputs.verbose)
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
        if self.inputs.test_split is not None and self.inputs.test_size > 0:
            r, r2 = self.crossval(X, Y, seed=self.rs,
                                  verbose=self.inputs.verbose)
            res.cvres.update(dict(pearson_r=r, r_squared=r2))

        # get rid of the stupid diagonal matrix
        res.s = np.diag(res.s)

        return res


# let's make it a function
def behavioral_pls(X, Y, *, groups=None, n_cond=1, n_perm=5000, n_boot=5000,
                   n_split=100, test_size=0.25, test_split=100,
                   covariance=False, rotate=True, ci=95, seed=None,
                   verbose=True, n_proc=None, **kwargs):
    pls = BehavioralPLS(X=X, Y=Y, groups=groups, n_cond=n_cond,
                        n_perm=n_perm, n_boot=n_boot, n_split=n_split,
                        test_size=test_size, test_split=test_split,
                        covariance=covariance, rotate=rotate, ci=ci, seed=seed,
                        verbose=verbose, n_proc=n_proc, **kwargs)
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
{cross_val}
{covariance}
{rotate}
{ci}
{proc_options}

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
