"""Tests for the base module."""

import numpy as np
import pytest
import pyls


# tests for gen_permsamp(), gen_bootsamp(), and gen_splitsamp() are all very
# similar because the code behind them is, in many senses, redundant.
# that being said, the differences between the functions are intricate enough
# that extracting the shared functionality would be more difficult than anyone
# has time for right now.
# thus, we get repetitive tests to make sure that nothing is screwed up!
def test_gen_permsamp():
    """Test the gen_permsamp function for generating permutation samples."""
    # test to make sure that there are no duplicates generated given a
    # sufficiently large number of samples / conditions to work with
    unique_perms = pyls.base.gen_permsamp([10, 10], 2, seed=1234, n_perm=10)
    assert unique_perms.shape == (40, 10)
    for n, perm in enumerate(unique_perms.T[::-1], 1):
        assert not (perm[:, None] == unique_perms[:, :-n]).all(axis=0).any()

    # test that random state works and gives equivalent permutations when
    # the same number of groups / conditions / permutations are provided
    same_perms = pyls.base.gen_permsamp([10, 10], 2, seed=1234, n_perm=10)
    assert same_perms.shape == (40, 10)
    assert np.all(unique_perms == same_perms)

    # test that, given a small number of samples and requesting a large number
    # of permutations, duplicate samples are given (and a warning is raised!)
    with pytest.warns(UserWarning):
        dupe_perms = pyls.base.gen_permsamp([2, 2], 1, n_perm=25)
    assert dupe_perms.shape == (4, 25)
    dupe = False
    for n, perm in enumerate(dupe_perms.T[::-1], 1):
        dupe = dupe or (perm[:, None] == dupe_perms[:, :-n]).all(axis=0).any()
    assert dupe

    # test that subject conditions are kept together during permutations
    # that is, each subject has two conditions so we want to make sure that
    # when we permute subject order both conditions for a given subject are
    # moved together
    cond_perms = pyls.base.gen_permsamp([10], 2, n_perm=10)
    assert cond_perms.shape == (20, 10)
    for n in range(10):
        comp = np.array([f + 10 if f < 10 else f - 10 for f in cond_perms[n]])
        assert np.all(comp == cond_perms[n + 10])

    # test that subjects are permuted between groups
    # that is, no permutation should result in a group having the same subjects
    group_perms = pyls.base.gen_permsamp([10, 10], 1, n_perm=10)
    g1, g2 = np.sort(group_perms[:10], 0), np.sort(group_perms[10:], 0)
    comp = np.arange(0, 10)[:, None]
    assert not np.any(np.all(comp == g1, axis=0))
    assert not np.any(np.all((comp + 10) == g2, axis=0))

    # test that permutations with groups and conditions are appropriate
    # we'll use unique_perms since that has 2 groups and 2 conditions already
    # we want to confirm that (1) subject conditions are permuted together, and
    # (2) subjects are permuted between groups
    g1, g2 = unique_perms[:20], unique_perms[20:]
    # confirm subject conditions are permuted together
    for g in [g1, g2]:
        for n in range(10):
            comp = [f + 10 if f < 10 or (f >= 20 and f < 30) else f - 10 for f in g[n]]
            assert np.all(comp == g[n + 10])
    # confirm subjects perare muted between groups
    comp = np.arange(0, 20)[:, None]
    assert not np.any(np.all(comp == np.sort(g1, axis=0), axis=0))
    assert not np.any(np.all((comp + 20) == np.sort(g2, axis=0), axis=0))


def test_gen_bootsamp():
    """Test the gen_bootsamp function for generating bootstrap samples."""
    # test to make sure that there are no duplicates generated given a
    # sufficiently large number of samples / conditions to work with
    unique_boots = pyls.base.gen_bootsamp([10, 10], 2, seed=1234, n_boot=10)
    assert unique_boots.shape == (40, 10)
    for n, perm in enumerate(unique_boots.T[::-1], 1):
        assert not (perm[:, None] == unique_boots[:, :-n]).all(axis=0).any()

    # test that random state works and gives equivalent bootstraps when
    # the same number of groups / conditions / bootstraps are provided
    same_boots = pyls.base.gen_bootsamp([10, 10], 2, seed=1234, n_boot=10)
    assert same_boots.shape == (40, 10)
    assert np.all(unique_boots == same_boots)

    # test that, given a small number of samples and requesting a large number
    # of bootstraps, duplicate samples are given (and a warning is raised!)
    with pytest.warns(UserWarning):
        dupe_boots = pyls.base.gen_bootsamp([5], 1, n_boot=125)
    assert dupe_boots.shape == (5, 125)
    dupe = False
    for n, perm in enumerate(dupe_boots.T[::-1], 1):
        dupe = dupe or (perm[:, None] == dupe_boots[:, :-n]).all(axis=0).any()
    assert dupe

    # test that bootstraps all have the minimum number of unique subjects
    # that is, since we are always bootstrapping within groups/conditions, we
    # want to ensure that there is never a case where e.g., an entire group is
    # replaced with ONE subject (unless there are only two subjects, but then
    # what are you really doing?)
    # we set a minumum subject threshold equal to 1/2 the number of samples in
    # the smallest group; thus, with e.g., groups of [10, 20, 30], the minimum
    # number of unique subjects in any given group for any given bootstrap
    # should be 5 (=10/2)
    for grp in np.split(unique_boots, 4, axis=0):
        for boot in grp.T:
            assert np.unique(boot).size >= 5

    # make sure that when we're resampling subjects we're doing it for all
    # conditions; this is a much easier check than for permutations!
    for n in range(10):
        assert np.all(unique_boots[n] + 10 == unique_boots[n + 10])
    for n in range(20, 30):
        assert np.all(unique_boots[n] + 10 == unique_boots[n + 10])


def test_gen_splitsamp():
    """Test the gen_splitsamp function for generating split samples."""
    # test to make sure that there are no duplicates generated given a
    # sufficiently large number of samples / conditions to work with
    unique_splits = pyls.base.gen_splits([10, 10], 2, seed=1234, n_split=10)
    assert unique_splits.shape == (40, 10)
    for n, perm in enumerate(unique_splits.T[::-1], 1):
        assert not (perm[:, None] == unique_splits[:, :-n]).all(axis=0).any()

    # test that random state works and gives equivalent splits when
    # the same number of groups / conditions / splits are provided
    same_splits = pyls.base.gen_splits([10, 10], 2, seed=1234, n_split=10)
    assert same_splits.shape == (40, 10)
    assert np.all(unique_splits == same_splits)

    # test that, given a small number of samples and requesting a large number
    # of splits, duplicate samples are given (and a warning is raised!)
    with pytest.warns(UserWarning):
        dupe_splits = pyls.base.gen_splits([5], 1, n_split=125)
    assert dupe_splits.shape == (5, 125)
    dupe = False
    for n, perm in enumerate(dupe_splits.T[::-1], 1):
        dupe = dupe or (perm[:, None] == dupe_splits[:, :-n]).all(axis=0).any()
    assert dupe

    # make sure that each group is split independently!
    for grp in np.split(unique_splits, 4, axis=0):
        assert np.all(np.sum(grp, axis=0) == 5)

    # make sure that `test_size` works as expected, too
    # `test_size` should determine the proportion of values set to False in
    # each group x condition
    # by default, `test_size` is 0.5, so the split is half-and-half, but if we
    # change it to e.g., 0.2, then there should be `0.2 * n_samples` False
    # values in each group x condition
    test_splits = pyls.base.gen_splits([10, 10], 2, n_split=10, test_size=0.2)
    for grp in np.split(test_splits, 4, axis=0):
        assert np.all(np.sum(grp, axis=0) == 8)


def test_BasePLS(pls_inputs):
    """Test the BasePLS base class functionality and input validation."""
    # test that BasePLS accepts all inputs and stores them correctly
    basepls = pyls.base.BasePLS(**pls_inputs)
    for key in pls_inputs.keys():
        assert hasattr(basepls.inputs, key)
        assert np.all(basepls.inputs[key] == pls_inputs[key])

    # test that groups are handled correctly
    X, n_samples = pls_inputs["X"], len(pls_inputs["X"])
    # when not provided, should be calculated
    basepls = pyls.base.BasePLS(X, n_cond=2)
    assert basepls.inputs.groups == [n_samples // 2]
    # when provided as an int, should be coerced into a list
    basepls = pyls.base.BasePLS(X, groups=n_samples // 2, n_cond=2)
    assert basepls.inputs.groups == [n_samples // 2]
    # when they don't match the number of samples in the input data, error
    with pytest.raises(ValueError):
        basepls = pyls.base.BasePLS(X, groups=[100, 100])

    # ensure errors are raised for not implemented
    with pytest.raises(NotImplementedError):
        basepls.gen_covcorr(pls_inputs["X"], pls_inputs["Y"])
    with pytest.raises(NotImplementedError):
        basepls.gen_distrib(pls_inputs["X"], pls_inputs["Y"])
