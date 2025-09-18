"""Tests for the utility functions."""

import numpy as np
from pyls import utils
import pytest
import tqdm


def test_empty_dict():
    """Test the _empty_dict utility function."""
    assert utils._empty_dict({})
    assert utils._empty_dict(dict())
    assert not utils._empty_dict(dict(d=10))
    assert not utils._empty_dict(dict(d=dict(d=dict(d=10))))
    assert not utils._empty_dict([])
    assert not utils._empty_dict(None)
    assert not utils._empty_dict("test")
    assert not utils._empty_dict(10)
    assert not utils._empty_dict(10.0)
    assert not utils._empty_dict(set())


def test_not_empty_keys():
    """Test the _not_empty_keys utility function."""
    assert utils._not_empty_keys(dict()) == set()
    assert utils._not_empty_keys(dict(test=10)) == {"test"}
    assert utils._not_empty_keys(dict(test=10, temp=None)) == {"test"}
    assert utils._not_empty_keys(dict(test=10, temp={})) == {"test"}

    with pytest.raises(TypeError):
        utils._not_empty_keys([10, 20, 30])


def test_ResDict():
    """Test the ResDict class."""

    # toy example with some allowed keys
    class TestDict(utils.ResDict):
        allowed = ["test", "temp"]

    # confirm string representations work
    d = utils.ResDict()
    assert str(d) == "ResDict()"
    assert str(TestDict(test={})) == "TestDict()"
    assert str(TestDict(test=None)) == "TestDict()"
    assert d != TestDict()

    # confirm general key checking works
    test1 = TestDict(test=10)
    test2 = TestDict(test=11)
    test3 = TestDict(test=10, temp=11)
    assert str(test1) == "TestDict(test)"
    assert str(test2) == "TestDict(test)"
    assert str(test3) == "TestDict(test, temp)"
    assert test1 == test1
    assert test1 != test2
    assert test1 != test3

    # confirm numpy array comparisons work
    test1 = TestDict(test=np.arange(9))
    test2 = TestDict(test=np.arange(9) + 1e-6)  # should work
    test3 = TestDict(test=np.arange(9) + 1e-5)  # too high
    test4 = TestDict(test=np.arange(10))  # totally different
    assert test1 == test1
    assert test1 == test2
    assert test1 != test3
    assert test1 != test4

    # confirm nested dictionary comparisons work
    test1 = TestDict(test=test1)
    test2 = TestDict(test=test3)
    assert test1 == test1
    assert test1 != test2

    # confirm item assignment holds
    test1.temp = 10
    assert test1.temp == 10
    assert test1 == test1
    assert test1 != test2

    # confirm rejection of item assignment not in cls.allowed
    test1.blargh = 10
    assert not hasattr(test1, "blargh")

    test1.temp = None
    test2.temp = None
    assert test1 != test2


def test_trange():
    """Test the trange utility function for progress bar functionality."""
    # test that verbose=False generates a range object
    out = utils.trange(1000, verbose=False, desc="Test tqdm")
    assert [f for f in out] == list(range(1000))
    # test that function will accept arbitrary kwargs and overwrite defaults
    out = utils.trange(1000, desc="Test tqdm", mininterval=0.5, ascii=False)
    assert isinstance(out, tqdm.tqdm)


def test_dummy_label():
    """Test the dummy_label utility function."""
    groups = [10, 12, 11]
    expected = [[10, 12, 11], [10, 10, 12, 12, 11, 11]]
    for n_cond in range(1, 3):
        dummy = utils.dummy_label(groups, n_cond=n_cond)
        assert dummy.shape == (np.sum(groups) * n_cond,)
        assert np.unique(dummy).size == len(groups) * n_cond
        for n, grp in enumerate(np.unique(dummy)):
            assert np.sum(dummy == grp) == expected[n_cond - 1][n]


def test_dummy_code():
    """Test the dummy_code utility function."""
    groups = [10, 12, 11]
    expected = [[10, 12, 11], [10, 10, 12, 12, 11, 11]]
    for n_cond in range(1, 3):
        dummy = utils.dummy_code(groups, n_cond=n_cond)
        assert dummy.shape == (np.sum(groups) * n_cond, len(groups) * n_cond)
        assert np.all(np.unique(dummy) == [0, 1])
        for n, grp in enumerate(dummy.T):
            assert grp.sum() == expected[n_cond - 1][n]


def test_permute_cols():
    """Test the permute_cols utility function."""
    x = np.arange(9).reshape(3, 3)
    expected = np.array([[0, 1, 5], [6, 4, 2], [3, 7, 8]])

    out = utils.permute_cols(x, seed=np.random.RandomState(1234))
    assert not np.all(out == x) and np.all(out == expected)

    # don't accept 1D arrays
    with pytest.raises(ValueError):
        utils.permute_cols(np.arange(9))


def test_unravel():
    """Test the _unravel context manager utility function."""
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert utils._unravel()(range(10)) == expected
    expected = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    assert utils._unravel()(x**2 for x in range(10)) == expected

    # test context manager status and arbitrary argument acceptance
    with utils._unravel(10, test=20) as cm:
        assert cm(x**2 for x in range(10)) == expected


def test_get_par_func():
    """Test the get_par_func utility function."""

    def fcn(x):
        return x

    assert fcn(10) == 10
    assert fcn([10, 10]) == [10, 10]

    if utils.joblib_avail:
        import joblib

        with utils.get_par_func(1000, fcn) as (par, func):
            assert isinstance(par, joblib.Parallel)
            assert par.n_jobs == 1000
            assert not fcn == func

        utils.joblib_avail = False
        with utils.get_par_func(1000, fcn) as (par, func):
            assert isinstance(par, utils._unravel)
            assert fcn == func
