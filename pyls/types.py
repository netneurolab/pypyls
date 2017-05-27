#!/usr/bin/env python

from pyls import compute


def behavioral_pls(X, Y, n_comp, n_perm=5000, n_boot=1000):
    """
    Runs behavioral PLS on `behamat` and `data`

    Parameters
    ----------
    X : array (N x k [x group])
    Y : array (N x j [x group])
    """

    U, d, V = compute.svd(X, Y, k=n_comp)

    if len(U) < len(V): orig = U
    else: orig = V

    perms = compute.serial_permute(X, Y,
                                   n_comp, orig,
                                   perms=n_perm)
    U_boot, V_boot = compute.bootstrap(X, Y,
                                       n_comp, U, V,
                                       boots=n_boot)
    U_bci, V_bci = compute.boot_ci(U_boot, V_boot)
    U_rel, V_rel = compute.boot_rel(U, V, U_boot, V_boot)

    pvals = compute.perm_sig(perms, d)
    d_sig = compute.kaiser_criterion(d)

    return U, d, V, pvals, d_sig, U_bci, V_bci, U_rel, V_rel
