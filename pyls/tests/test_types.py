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
groups = [33, 34, 33]

nosplit_attrs = ['inputs',
                 'U', 'd', 'V',
                 'd_pvals', 'd_varexp',
                 'U_bsr', 'V_bsr']
split_attrs = nosplit_attrs + ['U_corr', 'V_corr', 'U_pvals', 'V_pvals']


def test_BehavioralPLS_onegroup_onecond():
    # one group, one condition, without split-half resampling
    nosplit = pyls.types.BehavioralPLS(X, Y,
                                       n_perm=n_perm, n_boot=n_boot,
                                       n_split=None, seed=seed)
    # one group, one condition, with split-half resampling
    split = pyls.types.BehavioralPLS(X, Y,
                                     n_perm=n_perm, n_boot=n_boot,
                                     n_split=n_split, seed=seed)

    # ensure the outputs have appropriate attributes
    for f in nosplit_attrs:
        assert hasattr(nosplit, f)
    for f in split_attrs:
        assert hasattr(split, f)


def test_BehavioralPLS_multigroup_onecond():
    # multiple groups, one condition, without split-half resampling
    pyls.types.BehavioralPLS(X, Y, groups=groups,
                             n_perm=n_perm, n_boot=n_boot,
                             n_split=None, seed=seed)
    # multiple groups, one condition, with split-half resampling
    pyls.types.BehavioralPLS(X, Y, groups=groups,
                             n_perm=n_perm, n_boot=n_boot,
                             n_split=n_split, seed=seed)


def test_BehavioralPLS_onegroup_multicond():
    # one group, multiple conditions, without split-half resampling
    pyls.types.BehavioralPLS(X, Y, n_cond=4,
                             n_perm=n_perm, n_boot=n_boot,
                             n_split=None, seed=seed)
    # one group, multiple conditions, with split-half resampling
    pyls.types.BehavioralPLS(X, Y, n_cond=4,
                             n_perm=n_perm, n_boot=n_boot,
                             n_split=n_split, seed=seed)


def test_BehavioralPLS_multigroup_multicond():
    # multiple groups, multiple conditions, without split-half resampling
    pyls.types.BehavioralPLS(X, Y, groups=[25, 25], n_cond=2,
                             n_perm=n_perm, n_boot=n_boot,
                             n_split=None, seed=seed)
    # multiple groups, multiple conditions, with split-half resampling
    pyls.types.BehavioralPLS(X, Y, groups=[25, 25], n_cond=2,
                             n_perm=n_perm, n_boot=n_boot,
                             n_split=n_split, seed=seed)


def test_BehavioralPLS_errors():
    # confirm errors in cross-covariance matrix calculations
    with pytest.raises(ValueError):
        pyls.types.BehavioralPLS(Y[:, 0], X)
    with pytest.raises(ValueError):
        pyls.types.BehavioralPLS(Y[:, 0], X[:, 0])
    with pytest.raises(ValueError):
        pyls.types.BehavioralPLS(Y[:-1], X)
    with pytest.raises(ValueError):
        pyls.types.BehavioralPLS(X[:, :, None], Y[:, :, None])


def test_MeanCenteredPLS_multigroup_onecond():
    # multiple groups, one condition, without split-half resampling
    nosplit = pyls.types.MeanCenteredPLS(X, groups,
                                         n_perm=n_perm, n_boot=n_boot,
                                         n_split=None, seed=seed)
    # multiple groups, one condition, with split-half resampling
    split = pyls.types.MeanCenteredPLS(X, groups,
                                       n_perm=n_perm, n_boot=n_boot,
                                       n_split=n_split, seed=seed)

    # ensure the outputs have appropriate attributes
    for f in nosplit_attrs:
        assert hasattr(nosplit, f)
    for f in split_attrs:
        assert hasattr(split, f)


def test_MeanCenteredPLS_onegroup_multicond():
    # one group, multiple conditions, without split-half resampling
    pyls.types.MeanCenteredPLS(X, [subj], n_cond=2,
                               n_perm=n_perm, n_boot=n_boot,
                               n_split=None, seed=seed)
    # one group, multiple conditions, with split-half resampling
    pyls.types.MeanCenteredPLS(X, [subj], n_cond=2,
                               n_perm=n_perm, n_boot=n_boot,
                               n_split=n_split, seed=seed)


def test_MeanCenteredPLS_multigroup_multicond():
    # multiple groups, multiple conditions, without split-half resampling
    pyls.types.MeanCenteredPLS(X, groups=[25, 25], n_cond=2,
                               n_perm=n_perm, n_boot=n_boot,
                               n_split=None, seed=seed)
    # multiple groups, multiple conditions, with split-half resampling
    pyls.types.MeanCenteredPLS(X, groups=[25, 25], n_cond=2,
                               n_perm=n_perm, n_boot=n_boot,
                               n_split=n_split, seed=seed)
