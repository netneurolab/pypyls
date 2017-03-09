#!/usr/bin/env python

import pytest
import numpy as np
import pyls

brain      = 1000
behavior   = 100
comp       = 20
n_perms    = 500
n_boots    = 100

X = np.random.rand(comp,brain)
Y = np.random.rand(comp,behavior)

def test_pls():
    U, d, V = pyls.compute.svd(X, Y, comp, norm=True)
    assert d.shape == (comp,comp)
    assert U.shape == (behavior,comp)
    assert V.shape == (brain, comp)

    perms = pyls.compute.permute(X, Y, comp, U, perms=n_perms)
    assert perms.shape == (n_perms, comp)

    U_boot, V_boot = pyls.compute.bootstrap(X, Y, comp, U, V, boots=n_boots)
    assert U_boot.shape == (behavior, comp, n_boots)
    assert V_boot.shape == (brain, comp, n_boots)

    pvals = pyls.compute.perm_sig(perms, d)
    assert pvals.size == comp

    U_bci, V_bci = pyls.compute.boot_ci(U_boot, V_boot)
    assert U_bci.shape == (behavior, comp, 2)
    assert V_bci.shape == (brain, comp, 2)

    U_rel, V_rel = pyls.compute.boot_rel(U, V, U_boot, V_boot)
    assert U_rel.shape == (behavior, comp)
    assert V_rel.shape == (brain, comp)
