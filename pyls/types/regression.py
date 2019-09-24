# -*- coding: utf-8 -*-

import numpy as np
from ..base import BasePLS
from ..structures import _pls_input_docs
from .. import compute


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
    X0 = (X - X.mean(axis=0))
    Y0 = (Y - Y.mean(axis=0))
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
    for comp in range(n_components):
        ui = y_scores[:, [comp]]
        for repeat in range(2):
            for j in range(comp):
                tj = x_scores[:, [j]]
                ui = ui - ((tj.T @ ui) * tj)

        y_scores[:, [comp]] = ui

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
    def __init__(self, X, Y, *, n_components=None, n_perm=5000, n_boot=5000,
                 seed=None, verbose=True, n_proc=None, **kwargs):

        # check that inputs are valid
        if len(X) != len(Y):
            raise ValueError('Provided `X` and `Y` matrices must have the '
                             'same number of samples. Provided matrices '
                             'differed: X: {}, Y: {}'.format(len(X), len(Y)))

        max_components = min(len(X) - 1, X.shape[1])
        if n_components is None:
            n_components = max_components
        else:
            if not isinstance(n_components, int):
                raise ValueError('Provided `n_components` must be integer.')
            if n_components > max_components:
                raise ValueError('Provided `n_components` cannot be greater '
                                 'than {}'.format(max_components))

        super().__init__(X=np.asarray(X), Y=np.asarray(Y),
                         n_perm=n_perm, n_boot=n_boot,
                         n_split=0, test_split=0,
                         seed=seed, verbose=verbose,
                         n_proc=n_proc, **kwargs)

        # mean-center here so that our outputs are generated accordingly
        X = self.inputs.X - self.inputs.X.mean(axis=0, keepdims=True)
        Y = self.inputs.Y - self.inputs.Y.mean(axis=0, keepdims=True)
        self.n_components = n_components
        self.results = self.run_pls(X, Y, n_components)

    def make_permutation(self, X, Y, perminds):
        return X, Y[perminds]

    def _single_boot(self, X, Y, inds, groups=None, original=None, seed=None):
        res = simpls(X[inds], Y[inds], self.n_components, seed=seed)
        return None, res['x_weights']

    def _single_perm(self, X, Y, inds, groups=None, original=None, seed=None):
        Xp, Yp = self.make_permutation(X, Y, inds)
        res = simpls(Xp, Yp, self.n_components, seed=seed)['pctvar'][1]
        return res, None, None

    def svd(self, X, Y, groups=None, seed=None):
        res = simpls(X, Y, self.n_components, seed=seed)
        return res['x_weights'], res['pctvar'], None

    def run_pls(self, X, Y, n_components=None):
        res = super().run_pls(X, Y)

        if self.inputs.n_boot > 0:
            u_sum, u_square = self.bootstrap(X, Y, self.rs)[1:]
            u_sum, u_square = u_sum + res.u, u_square + (res.u ** 2)
            bsrs, uboot_se = compute.boot_rel(res.u, u_sum, u_square,
                                              self.inputs.n_boot + 1)
            res.bootres.update(dict(bootstrapratios=bsrs,
                                    uboot_se=uboot_se,
                                    bootsamples=self.bootsamp))
        return res


# let's make it a function
def pls_regression(X, Y, *, n_components=None, n_perm=5000, n_boot=5000,
                   seed=None, verbose=True, n_proc=None, **kwargs):
    pls = PLSRegression(X=X, Y=Y, n_components=n_components,
                        n_perm=n_perm, n_boot=n_boot,
                        seed=seed, verbose=verbose, n_proc=n_proc, **kwargs)
    return pls.results


pls_regression.__doc__ = r"""
Performs PLS Regression on `X` and `Y`.

PLS regression is a multivariate statistical approach that relates two sets
of variables together. Traditionally, one of these arrays
represents a set of brain features (e.g., functional connectivity
estimates) and the other represents a set of behavioral variables; however,
these arrays can be any two sets of features belonging to a common group of
samples.

Parameters
----------
{input_matrix}
Y : (S, T) array_like
    Input data matrix, where `S` is samples and `T` is features
{stat_test}
{proc_options}

Returns
----------
{pls_results}
""".format(**_pls_input_docs)
