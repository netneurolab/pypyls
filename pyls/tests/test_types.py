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

attrs = ['inputs',
         'U', 'd', 'V',
         'd_pvals', 'd_varexp',
         'U_bsr', 'V_bsr']


def test_BehavioralPLS():
    o1 = pyls.types.BehavioralPLS(X, Y,
                                  n_perm=n_perm, n_boot=n_boot,
                                  n_split=None, seed=seed)
    o2 = pyls.types.BehavioralPLS(Y, X,
                                  n_perm=n_perm, n_boot=n_boot,
                                  n_split=None, seed=seed+1)
    for f in attrs:
        assert hasattr(o1, f)

    with pytest.raises(ValueError):
        pyls.types.BehavioralPLS(Y[:, 0], X)
    with pytest.raises(ValueError):
        pyls.types.BehavioralPLS(Y[:, 0], X[:, 0])
    with pytest.raises(ValueError):
        pyls.types.BehavioralPLS(Y[:-1], X)
    with pytest.raises(ValueError):
        pyls.types.BehavioralPLS(X[:, :, None], Y[:, :, None])


def test_BehavioralPLS_groups():
    pyls.types.BehavioralPLS(X, Y, groups=groups,
                             n_perm=n_perm, n_boot=n_boot,
                             n_split=None,
                             seed=seed)


def test_BehavioralPLS_splithalf():
    split_attrs = ['U_corr', 'V_corr', 'U_pvals', 'V_pvals']

    o1 = pyls.types.BehavioralPLS(X, Y,
                                  n_perm=n_perm, n_boot=n_boot,
                                  n_split=n_split, seed=seed)
    o2 = pyls.types.BehavioralPLS(X, Y, groups=groups,
                                  n_perm=n_perm, n_boot=n_boot,
                                  n_split=n_split, seed=seed)
    for f in split_attrs:
        assert hasattr(o1, f)


def test_MeanCenteredPLS():
    o1 = pyls.types.MeanCenteredPLS(X, groups,
                                    n_perm=n_perm, n_boot=n_boot,
                                    n_split=None, seed=seed)
    o2 = pyls.types.MeanCenteredPLS(X, groups,
                                    n_perm=n_perm, n_boot=n_boot,
                                    n_split=n_split, seed=seed)
