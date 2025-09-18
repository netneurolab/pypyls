"""Behavioral PLS."""

import numpy as np
from sklearn.metrics import r2_score
from ..base import BasePLS, gen_splits
from ..structures import _pls_input_docs
from .. import compute, utils


class BehavioralPLS(BasePLS):
    """Behavioral PLS."""

    def __init__(
        self,
        X,
        Y,
        *,
        groups=None,
        n_cond=1,
        n_perm=5000,
        n_boot=5000,
        n_split=100,
        test_size=0.25,
        test_split=100,
        covariance=False,
        rotate=True,
        ci=95,
        permsamples=None,
        custom_permuted_Y=None,
        bootsamples=None,
        seed=None,
        verbose=True,
        n_proc=None,
        **kwargs,
    ):
        super().__init__(
            X=np.asarray(X),
            Y=np.asarray(Y),
            groups=groups,
            n_cond=n_cond,
            n_perm=n_perm,
            n_boot=n_boot,
            n_split=n_split,
            test_size=test_size,
            test_split=test_split,
            covariance=covariance,
            rotate=rotate,
            ci=ci,
            permsamples=permsamples,
            custom_permuted_Y=custom_permuted_Y,
            bootsamples=bootsamples,
            seed=seed,
            verbose=verbose,
            n_proc=n_proc,
            **kwargs,
        )

        self.results = self.run_pls(self.inputs.X, self.inputs.Y)

    def gen_covcorr(self, X, Y, groups, **kwargs):
        """
        Compute cross-covariance matrix from `X` and `Y`.

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
        return np.vstack(
            [
                compute.xcorr(X[grp], Y[grp], covariance=self.inputs.covariance)
                for grp in groups.T.astype(bool)
            ]
        )

    def gen_distrib(self, X, Y, original, groups, *args, **kwargs):
        """
        Find behavioral correlations for single bootstrap resample.

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        original : (B, L) array_like
            Left singular vectors from bootstrap
        groups : (S, J) array_like
            Dummy coded input array, where `S` is observations and `J`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.

        Returns
        -------
        distrib : (T, L)
            Behavioral correlations for single bootstrap resample
        """
        tusc = X @ compute.normalize(original)

        return self.gen_covcorr(tusc, Y, groups=groups)

    def crossval(self, X, Y, groups=None, seed=None):
        """
        Perform cross-validation of SVD of `X` and `Y`.

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None

        Returns
        -------
        r_scores : (C,) np.ndarray
            R (Pearon correlation) scores across train-test splits
        r2_scores : (C,) np.ndarray
            R^2 (coefficient of determination) scores across train-test splits
        """
        if groups is None:
            groups = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)

        # use gen_splits to handle grouping/condition vars in train/test split
        splits = gen_splits(
            self.inputs.groups,
            self.inputs.n_cond,
            self.inputs.test_split,
            seed=seed,
            test_size=self.inputs.test_size,
        )

        gen = utils.trange(
            self.inputs.test_split,
            verbose=self.inputs.verbose,
            desc="Running cross-validation",
        )
        with utils.get_par_func(
            self.inputs.n_proc, self.__class__._single_crossval
        ) as (par, func):
            out = par(
                func(self, X=X, Y=Y, inds=splits[:, i], groups=groups, seed=i)
                for i in gen
            )
        r_scores, r2_scores = [np.stack(o, axis=-1) for o in zip(*out)]

        return r_scores, r2_scores

    def _single_crossval(self, X, Y, inds, groups=None, seed=None):
        """
        Generate single cross-validated r and r^2 score.

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        inds : (S,) array_like
            Train-test split, where train = True and test = False
        groups : (S, J) array_like, optional
            Dummy coded input array, where `S` is observations and `J`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition. If not specified will be generated on-the-fly. Default:
            None
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None
        """
        if groups is None:
            groups = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)

        X_train, Y_train, dummy_train = X[inds], Y[inds], groups[inds]
        X_test, Y_test, dummy_test = X[~inds], Y[~inds], groups[~inds]
        # perform initial decomposition on train set
        U, d, V = self.svd(X_train, Y_train, groups=dummy_train, seed=seed)

        # rescale the test set based on the training set
        Y_pred = []
        for n, V_spl in enumerate(np.split(V, groups.shape[-1])):
            tr_grp = dummy_train[:, n].astype(bool)
            te_grp = dummy_test[:, n].astype(bool)
            rescaled = compute.rescale_test(
                X_train[tr_grp], X_test[te_grp], Y_train[tr_grp], U, V_spl
            )
            Y_pred.append(rescaled)
        Y_pred = np.vstack(Y_pred)

        # calculate r & r-squared from comp of rescaled test & true values
        r_scores = compute.efficient_corr(Y_test, Y_pred)
        r2_scores = r2_score(Y_test, Y_pred, multioutput="raw_values")

        return r_scores, r2_scores

    def run_pls(self, X, Y):
        """
        Run PLS analysis.

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        """
        res = super().run_pls(X, Y)

        # mechanism for splitting outputs along group / condition indices
        grps = np.repeat(res["inputs"]["groups"], res["inputs"]["n_cond"])
        res["y_scores"] = np.vstack(
            [
                y @ v
                for (y, v) in zip(
                    np.split(Y, np.cumsum(grps)[:-1]),
                    np.split(res["y_weights"], len(grps)),
                )
            ]
        )

        # get lvcorrs
        groups = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        res["y_loadings"] = self.gen_covcorr(res["x_scores"], Y, groups)

        if self.inputs.n_boot > 0:
            # compute bootstraps
            distrib, u_sum, u_square = self.bootstrap(X, Y, self.rs)

            # add original scaled singular vectors back in
            bs = res["x_weights"] @ res["singvals"]
            u_sum, u_square = u_sum + bs, u_square + (bs**2)

            # calculate bootstrap ratios and confidence intervals
            bsrs, uboot_se = compute.boot_rel(
                bs, u_sum, u_square, self.inputs.n_boot + 1
            )
            corrci = np.stack(compute.boot_ci(distrib, ci=self.inputs.ci), -1)

            # update results.boot_result dictionary
            res["bootres"].update(
                dict(
                    x_weights_normed=bsrs,
                    x_weights_stderr=uboot_se,
                    y_loadings=res["y_loadings"].copy(),
                    y_loadings_boot=distrib,
                    y_loadings_ci=corrci,
                    bootsamples=self.bootsamp,
                )
            )

        # compute cross-validated prediction-based metrics
        if self.inputs.test_split is not None and self.inputs.test_size > 0:
            r, r2 = self.crossval(X, Y, groups=self.dummy, seed=self.rs)
            res["cvres"].update(dict(pearson_r=r, r_squared=r2))

        # get rid of the stupid diagonal matrix
        res["varexp"] = np.diag(compute.varexp(res["singvals"]))
        res["singvals"] = np.diag(res["singvals"])

        return res


# let's make it a function
def behavioral_pls(  # noqa: D103
    X,
    Y,
    *,
    groups=None,
    n_cond=1,
    n_perm=5000,
    n_boot=5000,
    n_split=0,
    test_size=0.25,
    test_split=100,
    covariance=False,
    rotate=True,
    ci=95,
    permsamples=None,
    custom_permuted_Y=None,
    bootsamples=None,
    seed=None,
    verbose=True,
    n_proc=None,
    **kwargs,
):
    pls = BehavioralPLS(
        X=X,
        Y=Y,
        groups=groups,
        n_cond=n_cond,
        n_perm=n_perm,
        n_boot=n_boot,
        n_split=n_split,
        test_size=test_size,
        test_split=test_split,
        covariance=covariance,
        rotate=rotate,
        ci=ci,
        permsamples=permsamples,
        bootsamples=bootsamples,
        custom_permuted_Y=custom_permuted_Y,
        seed=seed,
        verbose=verbose,
        n_proc=n_proc,
        **kwargs,
    )
    return pls.results


behavioral_pls.__doc__ = r"""
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
{split_half}
{cross_val}
{covariance}
{rotate}
{ci}
{resamples}
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
