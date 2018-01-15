# -*- coding: utf-8 -*-

import numpy as np
import pytest
import pyls

brain = 1000
behavior = 100
subj = 100
n_perm = 50
n_boot = 10
n_split = 5
seed = 1234

np.random.rand(seed)
X = np.random.rand(subj, behavior)
Y = np.random.rand(subj, brain)
groups = np.hstack([[1] * int(np.ceil(subj / 2)),
                    [2] * int(np.floor(subj / 2))])
data = dict(X=X, Y=Y, groups=groups)
opts = dict(n_perm=n_perm,
            n_boot=n_boot,
            n_split=n_split,
            ci=95,
            n_proc=1,
            seed=seed)
inputs = {**data, **opts}

attrs = ['n_perm', 'n_boot', 'n_split',
         'ci', 'n_proc', 'seed', 'X', 'Y', 'groups']


def test_PLSInputs():
    pls_inputs = pyls.base.PLSInputs(**inputs)
    for key in attrs:
        assert hasattr(pls_inputs, key)
        assert np.all(getattr(pls_inputs, key) == inputs[key])


def test_BasePLS():
    basepls = pyls.base.BasePLS(**inputs)
    for key in inputs:
        assert hasattr(basepls.inputs, key)
        assert np.all(getattr(basepls.inputs, key) == inputs[key])
    with pytest.raises(NotImplementedError):
        basepls._gen_covcorr(**data)
    with pytest.raises(NotImplementedError):
        basepls._gen_permsamp(**data)
    with pytest.raises(NotImplementedError):
        basepls._gen_bootsamp(**data)
    with pytest.raises(NotImplementedError):
        basepls._gen_splits(**data)
    with pytest.raises(NotImplementedError):
        basepls._run_pls(**data)
    with pytest.raises(NotImplementedError):
        basepls._svd(**data)
    with pytest.raises(NotImplementedError):
        basepls._bootstrap(**data)
    with pytest.raises(NotImplementedError):
        basepls._permutation(**data)
    with pytest.raises(NotImplementedError):
        basepls._single_perm(**data)
    with pytest.raises(NotImplementedError):
        basepls._split_half(X, Y, X, Y, groups)
