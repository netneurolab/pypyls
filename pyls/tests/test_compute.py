#!/usr/bin/env python

import pytest
import numpy as np
import pyls

brain      = 1000
behavior   = 100
comp       = 20
n_perm     = 500
n_boot     = 100

X = np.random.rand(comp,brain)
Y = np.random.rand(comp,behavior)

def test_pls():
    U, d, V = pyls.compute.svd(X, Y, comp)
    assert d.shape == (comp,comp)
    assert U.shape == (behavior,comp)
    assert V.shape == (brain, comp)

    perms = pyls.compute.serial_permute(X, Y, comp, U, perms=n_perm)
    assert perms.shape == (n_perm, comp)
    pyls.compute.serial_permute(Y, X, comp, U, perms=n_perm)

    U_boot, V_boot = pyls.compute.bootstrap(X, Y, comp, U, V, boots=n_boot)
    assert U_boot.shape == (behavior, comp, n_boot)
    assert V_boot.shape == (brain, comp, n_boot)

    pvals = pyls.compute.perm_sig(perms, d)
    assert pvals.size == comp

    U_bci, V_bci = pyls.compute.boot_ci(U_boot, V_boot)
    assert U_bci.shape == (behavior, comp, 2)
    assert V_bci.shape == (brain, comp, 2)

    U_rel, V_rel = pyls.compute.boot_rel(U, V, U_boot, V_boot)
    assert U_rel.shape == (behavior, comp)
    assert V_rel.shape == (brain, comp)

    pyls.compute.boot_sig(U_bci[:,0,:])
    pyls.compute.kaiser_criterion(d)

    X[:,10], Y[:,10] = 0, 0
    U2, d2, V2 = pyls.compute.svd(X, Y, comp)
