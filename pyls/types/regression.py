# -*- coding: utf-8 -*-

import numpy as np
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
        Dictionary with x- and y-loadings / scores / weights / residuals,
        betas, percent variance explained, mse, and T2 values
    """

    X, Y = np.asanyarray(X), np.asanyarray(Y)
    if n_components is None:
        n_components = min(min(X.shape), min(Y.shape))

    X0 = (X - X.mean(axis=0))
    Y0 = (Y - Y.mean(axis=0))

    x_loadings = np.zeros((X.shape[1], n_components))
    y_loadings = np.zeros((Y.shape[1], n_components))
    x_scores = np.zeros((X.shape[0], n_components))
    y_scores = np.zeros((X.shape[0], n_components))
    x_weights = np.zeros((X.shape[1], n_components))
    y_weights = np.zeros((Y.shape[1], n_components))

    V = np.zeros((X.shape[1], n_components))

    Cov = X0.T @ Y0

    for comp in range(n_components):
        ci, si, ri = compute.svd(Cov, n_components=1, seed=seed)

        ti = X0 @ ri
        normti = np.linalg.norm(ti)
        ti /= normti
        x_loadings[:, [comp]] = X0.T @ ti

        qi = (si * ci) / normti
        y_loadings[:, [comp]] = qi

        x_scores[:, [comp]] = ti
        y_scores[:, [comp]] = Y0 @ qi

        x_weights[:, [comp]] = ri / normti
        y_weights[:, [comp]] = ci / np.linalg.norm(y_scores[:, [comp]])

        vi = x_loadings[:, [comp]]

        for repeat in range(2):
            for j in range(comp):
                vj = V[:, [j]]
                vi = vi - ((vj.T @ vi) * vj)
        vi /= np.linalg.norm(vi)
        V[:, [comp]] = vi

        Cov = Cov - (vi @ (vi.T @ Cov))
        Vi = V[:, :comp]
        Cov = Cov - (Vi @ (Vi.T @ Cov))

    for comp in range(n_components):
        ui = y_scores[:, [comp]]
        for repeat in range(2):
            for j in range(comp):
                tj = x_scores[:, [j]]
                ui = ui - ((tj.T @ ui) * tj)

        y_scores[:, [comp]] = ui

    beta = x_weights @ y_loadings.T
    beta = np.row_stack([Y.mean(0) - (X.mean(0) @ beta), beta])

    pctvar = [
        np.sum(np.abs(x_loadings) ** 2, 0) / np.sum(np.abs(X0)**2),
        np.sum(np.abs(y_loadings) ** 2, 0) / np.sum(np.abs(Y0)**2)
    ]

    mse = np.zeros((2, n_components + 1))
    mse[0, 0] = np.sum(np.abs(X0) ** 2)
    mse[1, 0] = np.sum(np.abs(Y0) ** 2)
    for i in range(n_components):
        X0_recon = x_scores[:, :i + 1] @ x_loadings[:, :i + 1].T
        Y0_recon = x_scores[:, :i + 1] @ y_loadings[:, :i + 1].T
        mse[0, i + 1] = np.sum(np.abs(X0 - X0_recon) ** 2)
        mse[1, i + 1] = np.sum(np.abs(Y0 - Y0_recon) ** 2)
    mse /= len(X)

    t2 = np.sum((np.abs(x_scores) ** 2) / np.var(x_scores, 0, ddof=1), 1)
    x_residuals = X0 - X0_recon
    y_residuals = Y0 - Y0_recon

    return dict(
        x_weights=x_weights,
        y_weights=y_weights,
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
