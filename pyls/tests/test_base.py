# -*- coding: utf-8 -*-

import numpy as np
import pytest
import pyls


n_cond = 1
n_perm = 50
n_boot = 10
n_split = 5
seed = 1234
groups = [50, 50]

np.random.rand(seed)
X = np.random.rand(100, 1000)
Y = np.random.rand(100, 100)

opts = dict(X=X, Y=Y,
            groups=groups, n_cond=n_cond,
            n_perm=n_perm, n_boot=n_boot, n_split=n_split,
            ci=95, n_proc=1, seed=seed)

attrs = ['X', 'Y', 'groups', 'n_cond',
         'n_perm', 'n_boot', 'n_split',
         'ci', 'n_proc', 'seed']


def test_PLSInputs():
    pls_inputs = pyls.base.PLSInputs(**opts)
    for key in attrs:
        assert hasattr(pls_inputs, key)
        assert np.all(getattr(pls_inputs, key) == opts[key])

    assert pyls.base.PLSInputs(n_split=0).n_split is None


def test_BasePLS():
    basepls = pyls.base.BasePLS(**opts)
    for key in attrs:
        assert hasattr(basepls.inputs, key)
        assert np.all(getattr(basepls.inputs, key) == opts[key])

    with pytest.raises(NotImplementedError):
        basepls.gen_covcorr(X, Y, groups)
