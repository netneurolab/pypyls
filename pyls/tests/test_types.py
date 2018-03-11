# -*- coding: utf-8 -*-

import numpy as np
import pytest
import pyls


brain = 1000
behavior = 100
subj = 100
n_perm = 20
n_boot = 10
n_split = 5
seed = 1234

np.random.rand(seed)
X = np.random.rand(subj, brain)
Y = np.random.rand(subj, behavior)


def make_outputs(behavior, num_lv):
    """
    Used to make list of expected attributes and shapes for PLS outputs

    Parameters
    ----------
    behavior : int
        Expected number of behavior output
    num_lv : int
        Expected number of output latent variables

    Returns
    -------
    nosplit_attrs : list-of-tuple
        For PLS without split-half resampling
    split_attrs : list-of-tuple
        For PLS with split-half resampling
    """

    nosplit_attrs = [
        ('U', (behavior, num_lv)),
        ('s', (num_lv, num_lv)),
        ('V', (brain, num_lv)),
        ('sp', (num_lv,)),
        ('sprob', (num_lv,)),
        ('d_varexp', (num_lv,)),
        ('U_bsr', (behavior, num_lv)),
        ('V_bsr', (brain, num_lv))
    ]

    split_attrs = [
        ('U_corr', (num_lv,)),
        ('V_corr', (num_lv,)),
        ('U_pvals', (num_lv,)),
        ('V_pvals', (num_lv,)),
    ] + nosplit_attrs

    return nosplit_attrs, split_attrs


def confirm_outputs(output, attributes):
    """
    Used to confirm ``output`` has expected ``attributes```

    Parameters
    ----------
    output : PLS output
    attributes : list-of-tuple
        From output of ``make_outputs()``
    """

    for (attr, shape) in attributes:
        assert hasattr(output, attr)
        assert getattr(output, attr).shape == shape


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


def test_BehavioralPLS_onegroup_onecond():
    # one group, one condition
    groups, n_cond = [subj], 1
    nosplit = pyls.types.BehavioralPLS(X, Y, groups=groups, n_cond=n_cond,
                                       n_perm=n_perm, n_boot=n_boot,
                                       n_split=None, seed=seed)
    split = pyls.types.BehavioralPLS(X, Y, groups=groups, n_cond=n_cond,
                                     n_perm=n_perm, n_boot=n_boot,
                                     n_split=n_split, seed=seed)

    # ensure the outputs have appropriate attributes
    nosplit_attrs, split_attrs = make_outputs(behavior * n_cond * len(groups),
                                              min([subj, brain, behavior]))
    confirm_outputs(nosplit, nosplit_attrs)
    confirm_outputs(split, split_attrs)


def test_BehavioralPLS_multigroup_onecond():
    # multiple groups, one condition
    groups, n_cond = [33, 34, 33], 1
    nosplit = pyls.types.BehavioralPLS(X, Y, groups=groups, n_cond=n_cond,
                                       n_perm=n_perm, n_boot=n_boot,
                                       n_split=None, seed=seed)
    split = pyls.types.BehavioralPLS(X, Y, groups=groups, n_cond=n_cond,
                                     n_perm=n_perm, n_boot=n_boot,
                                     n_split=n_split, seed=seed)

    nosplit_attrs, split_attrs = make_outputs(behavior * n_cond * len(groups),
                                              min([subj, brain, behavior]))
    confirm_outputs(nosplit, nosplit_attrs)
    confirm_outputs(split, split_attrs)


def test_BehavioralPLS_onegroup_multicond():
    # one group, multiple conditions
    groups, n_cond = [subj], 4
    nosplit = pyls.types.BehavioralPLS(X, Y, groups=groups, n_cond=n_cond,
                                       n_perm=n_perm, n_boot=n_boot,
                                       n_split=None, seed=seed)
    split = pyls.types.BehavioralPLS(X, Y, groups=groups, n_cond=n_cond,
                                     n_perm=n_perm, n_boot=n_boot,
                                     n_split=n_split, seed=seed)

    nosplit_attrs, split_attrs = make_outputs(behavior * n_cond * len(groups),
                                              min([subj, brain, behavior]))
    confirm_outputs(nosplit, nosplit_attrs)
    confirm_outputs(split, split_attrs)


def test_BehavioralPLS_multigroup_multicond():
    # multiple groups, multiple conditions
    groups, n_cond = [25, 25], 2
    nosplit = pyls.types.BehavioralPLS(X, Y, groups=groups, n_cond=n_cond,
                                       n_perm=n_perm, n_boot=n_boot,
                                       n_split=None, seed=seed)
    split = pyls.types.BehavioralPLS(X, Y, groups=groups, n_cond=n_cond,
                                     n_perm=n_perm, n_boot=n_boot,
                                     n_split=n_split, seed=seed)

    nosplit_attrs, split_attrs = make_outputs(behavior * n_cond * len(groups),
                                              min([subj, brain, behavior]))
    confirm_outputs(nosplit, nosplit_attrs)
    confirm_outputs(split, split_attrs)


def test_MeanCenteredPLS_multigroup_onecond():
    # multiple groups, one condition
    groups, n_cond = [33, 34, 33], 1
    nosplit = pyls.types.MeanCenteredPLS(X, groups, n_cond=n_cond,
                                         n_perm=n_perm, n_boot=n_boot,
                                         n_split=None, seed=seed)
    split = pyls.types.MeanCenteredPLS(X, groups, n_cond=n_cond,
                                       n_perm=n_perm, n_boot=n_boot,
                                       n_split=n_split, seed=seed)

    nosplit_attrs, split_attrs = make_outputs(len(groups) * n_cond,
                                              (len(groups) * n_cond))
    confirm_outputs(nosplit, nosplit_attrs)
    confirm_outputs(split, split_attrs)


def test_MeanCenteredPLS_onegroup_multicond():
    # one group, multiple conditions
    groups, n_cond = [subj], 2
    nosplit = pyls.types.MeanCenteredPLS(X, groups, n_cond=n_cond,
                                         n_perm=n_perm, n_boot=n_boot,
                                         n_split=None, seed=seed)
    split = pyls.types.MeanCenteredPLS(X, groups, n_cond=n_cond,
                                       n_perm=n_perm, n_boot=n_boot,
                                       n_split=n_split, seed=seed)

    nosplit_attrs, split_attrs = make_outputs(len(groups) * n_cond,
                                              (len(groups) * n_cond))
    confirm_outputs(nosplit, nosplit_attrs)
    confirm_outputs(split, split_attrs)


def test_MeanCenteredPLS_multigroup_multicond():
    # multiple groups, multiple conditions
    groups, n_cond = [25, 25], 2
    nosplit = pyls.types.MeanCenteredPLS(X, groups=groups, n_cond=n_cond,
                                         n_perm=n_perm, n_boot=n_boot,
                                         n_split=None, seed=seed)
    split = pyls.types.MeanCenteredPLS(X, groups=groups, n_cond=n_cond,
                                       n_perm=n_perm, n_boot=n_boot,
                                       n_split=n_split, seed=seed)

    nosplit_attrs, split_attrs = make_outputs(len(groups) * n_cond,
                                              (len(groups) * n_cond))
    confirm_outputs(nosplit, nosplit_attrs)
    confirm_outputs(split, split_attrs)
