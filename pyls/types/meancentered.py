"""Mean-centered PLS."""

import warnings
import numpy as np
from ..base import BasePLS
from ..structures import _pls_input_docs
from .. import compute, utils


class MeanCenteredPLS(BasePLS):
    """Mean-centered PLS."""

    def __init__(
        self,
        X,
        groups=None,
        n_cond=1,
        mean_centering=0,
        n_perm=5000,
        n_boot=5000,
        n_split=100,
        rotate=True,
        ci=95,
        permsamples=None,
        bootsamples=None,
        seed=None,
        verbose=True,
        n_proc=None,
        **kwargs,
    ):
        # check that groups and conditions are set appropriately
        if groups is None:
            if len(X) // n_cond != len(X) / n_cond:
                raise ValueError(
                    "Provided `X` matrix with {} samples is not "
                    "evenly divisible into {} conditions. Please "
                    "confirm inputs are correct and try again. ".format(len(X), n_cond)
                )
            groups = [len(X) // n_cond]
        elif not isinstance(groups, (list, np.ndarray)):
            groups = [groups]

        # check inputs for validity
        if n_cond == 1 and len(groups) == 1:
            raise ValueError(
                "Cannot perform PLS with only one group and one "
                "condition. Please confirm inputs are correct."
            )
        if n_cond == 1 and mean_centering == 0:
            warnings.warn(
                "Cannot set mean_centering to 0 when there is only "
                "one condition. Resetting mean_centering to 1.",
                stacklevel=2,
            )
            mean_centering = 1
        elif len(groups) == 1 and mean_centering == 1:
            warnings.warn(
                "Cannot set mean_centering to 1 when there is only "
                "one group. Resetting mean_centering to 0.",
                stacklevel=2,
            )
            mean_centering = 0

        # instantiate base class, generate dummy array, and run PLS analysis
        super().__init__(
            X=np.asarray(X),
            groups=groups,
            n_cond=n_cond,
            mean_centering=mean_centering,
            n_perm=n_perm,
            n_boot=n_boot,
            n_split=n_split,
            rotate=rotate,
            ci=ci,
            permsamples=permsamples,
            bootsamples=bootsamples,
            seed=seed,
            verbose=verbose,
            n_proc=n_proc,
            **kwargs,
        )
        self.inputs.Y = utils.dummy_code(self.inputs.groups, self.inputs.n_cond)
        self.results = self.run_pls(self.inputs.X, self.inputs.Y)

    def gen_covcorr(self, X, Y, **kwargs):
        """
        Compute mean-centered matrix from `X` and `Y`.

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
        mean_centered = compute.get_mean_center(
            X, Y, self.inputs.n_cond, self.inputs.mean_centering, means=True
        )
        return mean_centered

    def gen_distrib(self, X, Y, original, *args, **kwargs):
        """
        Find contrast for single bootstrap resample.

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Dummy coded input array, where `S` is observations and `T`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.
        original : (B, L) array_like
            Left singular vectors from bootstrap

        Returns
        -------
        distrib : (T, L)
            Contrast for single bootstrap resample
        """
        usc = compute.get_mean_center(
            X, Y, self.inputs.n_cond, self.inputs.mean_centering, means=False
        )
        usc = usc @ compute.normalize(original)

        return np.vstack([usc[g].mean(axis=0) for g in Y.T.astype(bool)])

    def make_permutation(self, X, Y, perminds):
        """
        Permute `X` according to `perminds`, leaving `Y` un-permuted.

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        perminds : (S,) array_like
            Array by which to permute `X`

        Returns
        -------
        Xp : (S, B) array_like
            `X`, permuted according to `perminds`
        Yp : (S, T) array_like
            Identical to `Y`
        """
        return X[perminds], Y

    def run_pls(self, X, Y):
        """
        Run PLS analysis.

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like, optional
            Dummy coded input array, where `S` is observations and `T`
            corresponds to the number of different groups x conditions. A value
            of 1 indicates that an observation belongs to a specific group or
            condition.

        Returns
        -------
        res : :obj:`pyls.structures.PLSResults`
            PLS results object
        """
        res = super().run_pls(X, Y)
        res["y_scores"] = Y @ res["y_weights"]

        # get normalized brain scores and contrast
        brainscores_dm = (
            compute.get_mean_center(
                X, Y, self.inputs.n_cond, self.inputs.mean_centering, False
            )
            @ res["x_weights"]
        )
        contrast = np.vstack(
            [brainscores_dm[grp].mean(axis=0) for grp in Y.T.astype(bool)]
        )

        if self.inputs.n_boot > 0:
            # compute bootstraps
            distrib, u_sum, u_square = self.bootstrap(X, Y, self.rs)

            # calculate bootstrap ratios and confidence intervals
            bs = res["x_weights"] @ res["singvals"]
            bsrs, uboot_se = compute.boot_rel(bs, u_sum, u_square, self.inputs.n_boot)
            corrci = np.stack(compute.boot_ci(distrib, ci=self.inputs.ci), -1)

            # update results.boot_result dictionary
            res["bootres"].update(
                dict(
                    x_weights_normed=bsrs,
                    x_weights_stderr=uboot_se,
                    bootsamples=self.bootsamp,
                    contrast=contrast,
                    contrast_boot=distrib,
                    contrast_ci=corrci,
                )
            )

        # get rid of the stupid diagonal matrix
        res["varexp"] = np.diag(compute.varexp(res["singvals"]))
        res["singvals"] = np.diag(res["singvals"])

        return res


def meancentered_pls(  # noqa: D103
    X,
    *,
    groups=None,
    n_cond=1,
    mean_centering=0,
    n_perm=5000,
    n_boot=5000,
    n_split=0,
    rotate=True,
    ci=95,
    permsamples=None,
    bootsamples=None,
    seed=None,
    verbose=True,
    n_proc=None,
    **kwargs,
):
    pls = MeanCenteredPLS(
        X=X,
        groups=groups,
        n_cond=n_cond,
        mean_centering=mean_centering,
        n_perm=n_perm,
        n_boot=n_boot,
        n_split=n_split,
        rotate=rotate,
        ci=ci,
        permsamples=permsamples,
        bootsamples=bootsamples,
        seed=seed,
        verbose=verbose,
        n_proc=n_proc,
        **kwargs,
    )
    return pls.results


meancentered_pls.__doc__ = r"""
Performs mean-centered PLS on `X`, sorted into `groups` and `conditions`.

Mean-centered PLS is a multivariate statistical approach that attempts to
find sets of variables in a matrix which maximally discriminate between
subgroups within the matrix.

While it carries the name PLS, mean-centered PLS is perhaps more related to
principal components analysis than it is to :obj:`pyls.behavioral_pls`. In
contrast to behavioral PLS, mean-centered PLS does not construct a cross-
covariance matrix. Instead, it operates by averaging the provided data
(`X`) within groups and/or conditions. The resultant matrix :math:`M` is
mean-centered, generating a new matrix :math:`R_{{mean\_centered}}` which
is submitted to singular value decomposition.

Parameters
----------
{input_matrix}
{groups}
{conditions}
{mean_centering}
{stat_test}
{split_half}
{rotate}
{ci}
{resamples}
{proc_options}

Returns
----------
{pls_results}

Notes
-----
The provided `mean_centering` argument can be changed to highlight or
"boost" potential group / condition differences by modifying how
:math:`R_{{mean\_centered}}` is generated:

- `mean_centering=0` will remove group means collapsed across conditions,
  emphasizing potential differences between conditions while removing
  overall group differences
- `mean_centering=1` will remove condition means collapsed across groups,
  emphasizing potential differences between groups while removing overall
  condition differences
- `mean_centering=2` will remove the grand mean collapsed across both
  groups _and_ conditions, permitting investigation of the full spectrum of
  potential group and condition effects.

{decomposition_narrative}

References
----------
{references}
""".format(**_pls_input_docs)
