# -*- coding: utf-8 -*-

import numpy as np
from ..base import BasePLS, gen_bootsamp
from ..structures import _pls_input_docs
from .. import compute


def resid_yscores(x_scores, y_scores, copy=True):
    """
    Orthogonalizes `y_scores` with respect to preceding `x_scores`

    Residualizes each column of `y_scores` against all previous columns of
    `x_scores` such that the column represents only the "new" contributions of
    each PLS component

    Parameters
    ----------
    x_scores : (S, L) array_like
        Projections of X data matrix into PLS-derived component space
    y_scores : (S, L) array_like
        Projections of Y data matrix into PLS-derived component space
    copy : bool, optional
        Whether to copy `y_scores` instead of overwriting in-place. Default:
        True

    Returns
    -------
    y_scores : (S, L) numpy.ndarray
        Residualized `y_scores`
    """

    x_scores = np.array(x_scores)
    y_scores = np.array(y_scores, copy=copy)

    for comp in range(x_scores.shape[1]):
        ui = y_scores[:, [comp]]
        for _ in range(2):
            for j in range(comp):
                tj = x_scores[:, [j]]
                ui = ui - ((tj.T @ ui) * tj)

        y_scores[:, [comp]] = ui

    return y_scores


def get_mask(X, Y):
    """ Returns mask removing rows where either X/Y contain all NaN values
    """

    return np.logical_not(np.logical_or(np.all(np.isnan(X), axis=1),
                                        np.all(np.isnan(Y), axis=1)))


def simpls(X, Y, n_components=None, seed=1234):
    """
    Performs partial least squares regression with the SIMPLS algorithm

    Parameters
    ----------
    X : (S, B) array_like
        Input data matrix, where `S` is observations and `B` is features
    Y : (S, T) array_like
        Input data matrix, where `S` is observations and `T` is features
    n_components : int, optional
        Number of components to estimate. If not specified will use the
        smallest of the input X and Y dimensions. Default: None
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed to use for random number generation. Helps ensure reproducibility
        of results. Default: None

    Returns
    -------
    results : dict
        Dictionary with x- and y-loadings / scores / residuals, x-weights,
        betas, percent variance explained, mse, and T2 values

    References
    ----------
    https://www.mathworks.com/help/stats/plsregress.html
    """

    X, Y = np.asanyarray(X), np.asanyarray(Y)
    if n_components is None:
        n_components = min(len(X) - 1, X.shape[1])

    # center variables and calculate covariance matrix
    X0 = (X - X.mean(axis=0, keepdims=True))
    Y0 = (Y - Y.mean(axis=0, keepdims=True))
    Cov = X0.T @ Y0

    # to store outputs
    x_loadings = np.zeros((X.shape[1], n_components))
    y_loadings = np.zeros((Y.shape[1], n_components))
    x_scores = np.zeros((X.shape[0], n_components))
    y_scores = np.zeros((X.shape[0], n_components))
    x_weights = np.zeros((X.shape[1], n_components))
    V = np.zeros((X.shape[1], n_components))

    for comp in range(n_components):
        # get first component of SVD of covariance matrix
        ci, si, ri = compute.svd(Cov, n_components=1, seed=seed)

        ti = X0 @ ri
        normti = np.linalg.norm(ti)

        # rescale such that:
        #     np.diag(x_weights.T @ X0.T @ X0 @ x_weights)
        #     == np.diag(x_scores.T @ x_scores)
        #     == 1
        x_weights[:, [comp]] = ri / normti

        # rescale such that np.diag(x_scores.T @ x_scores) == 1
        ti /= normti
        x_scores[:, [comp]] = ti

        x_loadings[:, [comp]] = X0.T @ ti  # == X0.T @ X0 @ x_weights
        qi = Y0.T @ ti
        y_loadings[:, [comp]] = qi
        y_scores[:, [comp]] = Y0 @ qi  # == Y0 @ Y0.T @ x_scores

        # update the orthonormal basis with modified Gram Schmidt; repeat twice
        # for additional stability
        vi = x_loadings[:, [comp]]
        for repeat in range(2):
            for j in range(comp):
                vj = V[:, [j]]
                vi = vi - ((vj.T @ vi) * vj)
        vi /= np.linalg.norm(vi)
        V[:, [comp]] = vi

        # deflate Cov, i.e. project onto ortho-complement of the x_loadings.
        # first, remove projections along the current basis vector, then remove
        # any component along previous basis vectors that's crept in as noise
        # from previous deflations.
        Cov = Cov - (vi @ (vi.T @ Cov))
        Vi = V[:, :comp]
        Cov = Cov - (Vi @ (Vi.T @ Cov))

    # by convention, orthogonalize the y_scores w.r.t. the preceding x_scores,
    # i.e. x_scores.T @ y_scores will be lower triangular. this gives, in
    # effect, only the "new" contribution to the y_scores for each PLS
    # component. it is also consistent with the PLS-1/PLS-2 algorithms, where
    # the y_scores are computed as linear combinations of a successively-
    # deflated Y0. use modified Gram-Schmidt, repeated twice for stability.
    y_scores = resid_yscores(x_scores, y_scores)

    # calculate betas and add intercept
    beta = x_weights @ y_loadings.T
    beta = np.row_stack([Y.mean(axis=0) - (X.mean(axis=0) @ beta), beta])

    # percent variance explained for both X and Y for all components
    pctvar = [
        np.sum(x_loadings ** 2, axis=0) / np.sum(X0 ** 2),
        np.sum(y_loadings ** 2, axis=0) / np.sum(Y0 ** 2)
    ]

    # mean squared error for models
    mse = np.zeros((2, n_components + 1))
    mse[0, 0] = np.sum(np.abs(X0) ** 2)
    mse[1, 0] = np.sum(np.abs(Y0) ** 2)
    for i in range(n_components):
        X0_recon = x_scores[:, :i + 1] @ x_loadings[:, :i + 1].T
        Y0_recon = x_scores[:, :i + 1] @ y_loadings[:, :i + 1].T
        mse[0, i + 1] = np.sum(np.abs(X0 - X0_recon) ** 2)
        mse[1, i + 1] = np.sum(np.abs(Y0 - Y0_recon) ** 2)
    mse /= len(X)

    t2 = np.sum((np.abs(x_scores) ** 2) / np.var(x_scores, axis=0, ddof=1), 1)
    x_residuals = X0 - X0_recon
    y_residuals = Y0 - Y0_recon

    return dict(
        x_weights=x_weights,
        x_loadings=x_loadings,
        y_loadings=y_loadings,
        x_scores=x_scores,
        y_scores=y_scores,
        x_residuals=x_residuals,
        y_residuals=y_residuals,
        beta=beta,
        pctvar=pctvar,
        mse=mse,
        t2=t2,
    )


class PLSRegression(BasePLS):
    def __init__(self, X, Y, *, n_components=None,
                 n_perm=5000, n_boot=5000,
                 rotate=True, ci=95, aggfunc='mean',
                 permsamples=None, bootsamples=None,
                 seed=None, verbose=True, n_proc=None,
                 **kwargs):

        # check n_components isn't unreasonable
        max_components = min(len(X) - 1, X.shape[1])
        if n_components is None:
            n_components = max_components
        else:
            n_components = int(n_components)
            if n_components > max_components:
                raise ValueError('Provided `n_components` cannot be greater '
                                 'than {}'.format(max_components))

        # bootstrapping is more complicated in this instance
        if Y.ndim == 3:
            if bootsamples is None:
                # we can generate exactly what we need for the bootsamples
                s = gen_bootsamp([Y.shape[0]], n_cond=1, n_boot=n_boot,
                                 seed=seed, verbose=verbose)
                c = gen_bootsamp([Y.shape[-1]], n_cond=1, n_boot=n_boot,
                                 seed=seed, verbose=verbose)
                bootsamples = np.array(list(zip(s.T, c.T))).T
            else:
                # we expect a (2, n_boot) array, where the first row is an
                # array of arrays, each of which is the size of the first dim
                # of `Y`, and the second row is an array of arrays, each of
                # which is the size of the third dim of `Y`
                bootsamples = np.asarray(bootsamples)
                s, c = [np.row_stack(b).shape[-1] for b in bootsamples]
                sexp, cexp = Y.shape[0], Y.shape[-1]
                if bootsamples.shape != (2, n_boot) or s != sexp or c != cexp:
                    raise ValueError('Provided bootsamples arrays does not '
                                     'match size of provided input arrays or '
                                     'number of bootstraps requested via '
                                     '`nboot`.')

            # also, we only care about aggfunc if we have a 3d `Y` matrix
            aggfuncs = dict(mean=np.mean, median=np.median, sum=np.sum)
            if not callable(aggfunc) and aggfunc not in aggfuncs:
                raise ValueError('Provided `aggfunc` must either be callable '
                                 'or one of {}'.format(sorted(aggfuncs)))
            self.aggfunc = aggfuncs.get(aggfunc, aggfunc)

        # these need to be zero -- they're not implemented for PLSRegression
        kwargs.update(n_split=0, test_split=0)
        super().__init__(X=np.asarray(X), Y=np.asarray(Y),
                         n_components=n_components, n_perm=n_perm,
                         n_boot=n_boot, rotate=rotate, ci=ci, aggfunc=aggfunc,
                         permsamples=permsamples, bootsamples=bootsamples,
                         seed=seed, verbose=verbose, n_proc=n_proc, **kwargs)

        self.n_components = n_components
        self.results = self.run_pls(self.inputs.X, self.inputs.Y)

    def svd(self, X, Y, seed=None):
        """
        Runs PLS decomposition with `X` and `Y`

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
        x_weights : (B, L) numpy.ndarray
            Weights of `B` features used to project `X` into PLS-derived
            component space
        varexp : (L, L) numpy.ndarray
            Variance explained by PLS-derived components; diagonal array
        """

        # find nan rows and mask for the decomposition
        mask = get_mask(X, Y)
        out = simpls(X[mask], Y[mask], self.n_components, seed=seed)

        # need to return len-3 for compatibility purposes
        # use the variance explained in Y in lieu of the singular values since
        # that's what we'll be testing against in permutations
        return out['x_weights'], np.diag(out['pctvar'][1]), None

    def _single_boot(self, X, Y, inds, groups=None, original=None, seed=None):
        """
        Bootstraps `X` and `Y` (w/replacement) and recomputes PLS decomposition

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        inds : (S,) array_like
            Resampling array
        groups : None
            Do nothing; here for compatibility purposes
        original : (B, N) array_like
            Weights of `X` from original (non-bootstrapped) decomposition; used
            to align bootstrapped weights
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None

        Returns
        -------
        y_loadings : (T, L) np.ndarray
            Loadings of `Y` on PLS decomposition from resampled data
        x_weights : (B, L) np.ndarray
            Weights of `X` from PLS decomposition of resampled data
        """

        # if we have a 3d `Y` matrix then our bootstrap matrix is complicated
        if Y.ndim == 3:
            sboot, cboot = inds
            Xi, Yi = X[sboot], self.aggfunc(Y[..., cboot], axis=-1)[sboot]
        # otherwise, very normal easy bootstrap
        else:
            Xi, Yi = X[inds], Y[inds]

        x_weights = self.svd(Xi, Yi, seed=seed)[0]

        if original is not None:
            # flip signs of weights based on correlations with `original`
            flip = np.sign(compute.efficient_corr(x_weights, original))
            x_weights *= flip
            # NOTE: should we be doing a procrustes here?

        # compute y_loadings
        mask = get_mask(Xi, Yi)
        y_loadings = Yi[mask].T @ (Xi @ x_weights)[mask]

        return y_loadings, x_weights

    def _single_perm(self, X, Y, inds, groups=None, original=None, seed=None):
        """
        Permutes `Y` (w/o replacement) and recomputes PLS decomposition

        Parameters
        ----------
        X : (S, B) array_like
            Input data matrix, where `S` is observations and `B` is features
        Y : (S, T) array_like
            Input data matrix, where `S` is observations and `T` is features
        inds : (S,) array_like
            Resampling array
        groups : None
            Do nothing; here for compatibility purposes
        original : (B, N) array_like, optional
            Weights of `X` from original (non-permuted) decomposition; used to
            to align permuted weights
        seed : {int, :obj:`numpy.random.RandomState`, None}, optional
            Seed for random number generation. Default: None

        Returns
        -------
        varexp : (L,) `numpy.ndarray`
            Variance explained by PLS decomposition of permuted data
        """

        # should permute Y (but not X) by default
        Xp, Yp = self.make_permutation(X, Y, inds)
        x_weights, varexp, _ = self.svd(Xp, Yp, seed=seed)

        if self.inputs.rotate and original is not None:
            # flip signs of weights based on correlations with `original`
            flip = np.sign(compute.efficient_corr(x_weights, original))
            x_weights *= flip
            # NOTE: should we be doing a procrustes here?

            # recompute pctvar based on new x_weight signs
            mask = get_mask(Xp, Yp)
            y_loadings = Yp[mask].T @ (Xp @ x_weights)[mask]
            varexp = np.sum(y_loadings ** 2, axis=0) / np.sum(Yp[mask] ** 2)
        else:
            varexp = np.diag(varexp)

        # need to return len-3 for compatibility purposes
        return varexp, None, None

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

        try:
            Y_agg = self.aggfunc(Y, axis=-1) if Y.ndim == 3 else Y
        except TypeError:
            raise TypeError('Provided callable `aggfun` must accept `axis` '
                            'keyword argument to condense an array along '
                            'the specified axis.')

        # mean-center here so that our outputs are generated accordingly
        X -= np.nanmean(X, axis=0, keepdims=True)
        Y_agg -= np.nanmean(Y_agg, axis=0, keepdims=True)
        mask = get_mask(X, Y_agg)

        res = super().run_pls(X, Y_agg)
        res['y_loadings'] = Y_agg[mask].T @ res['x_scores'][mask]
        res['y_scores'] = np.full((len(Y_agg), self.n_components), np.nan)
        res['y_scores'][mask] = resid_yscores(res['x_scores'][mask],
                                              Y_agg[mask] @ res['y_loadings'])

        if self.inputs.n_boot > 0:
            # compute bootstraps
            distrib, u_sum, u_square = self.bootstrap(X, Y, self.rs)

            # add original weights back in so we account for those
            bs = res['x_weights']
            u_sum, u_square = u_sum + bs, u_square + (bs ** 2)

            # calculate normalized ratios + bootstrap errors
            bsrs, uboot_se = compute.boot_rel(bs, u_sum, u_square,
                                              self.inputs.n_boot + 1)
            corrci = np.stack(compute.boot_ci(distrib, ci=self.inputs.ci), -1)
            res['bootres'].update(dict(x_weights_normed=bsrs,
                                       x_weights_stderr=uboot_se,
                                       y_loadings=res['y_loadings'],
                                       y_loadings_boot=distrib,
                                       y_loadings_ci=corrci,
                                       bootsamples=self.bootsamp,))

        # don't keep this as a diagonal matrix
        res['varexp'] = np.diag(res['singvals'])
        del res['singvals']

        return res


# let's make it a function
def pls_regression(X, Y, *, n_components=None, n_perm=5000, n_boot=5000,
                   rotate=True, ci=95, aggfunc='mean',
                   permsamples=None, bootsamples=None,
                   seed=None, verbose=True, n_proc=None, **kwargs):
    pls = PLSRegression(X=X, Y=Y, n_components=n_components, n_perm=n_perm,
                        n_boot=n_boot, rotate=rotate, ci=ci, aggfunc=aggfunc,
                        permsamples=permsamples, bootsamples=bootsamples,
                        seed=seed, verbose=verbose, n_proc=n_proc, **kwargs)
    return pls.results


pls_regression.__doc__ = r"""
Performs PLS regression on `X` and `Y`

PLS regression is a multivariate statistical approach that relates two sets
of variables together. Traditionally, one of these arrays
represents a set of brain features (e.g., functional connectivity
estimates) and the other represents a set of behavioral variables; however,
these arrays can be any two sets of features belonging to a common group of
samples.

This implementation of PLS regression uses the SIMPLS algorithm from [R1]_.

Parameters
----------
{input_matrix}
Y : (S, T[, C]) array_like
    Input data matrix, where `S` is samples and `T` is features. A 3d array
    can optionally be provided, where `C` indicates separate observations. In
    this case, bootstrapping will be performed over the final axis (`C`) and
    and the array will be collapsed with `aggfunc` prior to decomposition.
n_components : int, optional
    Number of components to estimate. If not specified this will be set to
    min(`S-1`, `B`). Default: None
{stat_test}
{rotate}
{ci}
aggfunc : str or callable, optional
    If `Y` is provided as a 3D array then this function will be used to reduce
    the final axis of the matrix. If a string is provided it must be one of
    ['mean', 'median', 'sum']. Default: 'mean'
{resamples}
{proc_options}

Returns
----------
{pls_results}

References
----------
.. [R1] De Jong, S. (1993). SIMPLS: an alternative approach to partial least
   squares regression. Chemometrics and intelligent laboratory systems, 18(3),
   251-263.
.. [R2] Rosipal, R., & Krämer, N. (2005, February). Overview and recent
   advances in partial least squares. In International Statistical and
   Optimization Perspectives Workshop" Subspace, Latent Structure and Feature
   Selection" (pp. 34-51). Springer, Berlin, Heidelberg.
.. [R3] https://www.mathworks.com/help/stats/plsregress.html
""".format(**_pls_input_docs)
