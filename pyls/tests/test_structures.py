"""Tests for the structures module."""

import multiprocessing as mp
import numpy as np
import pytest
from pyls import structures


def test_PLSInputs(pls_inputs):
    """Test PLSInputs structure initialization and validation."""
    # check correct handling of all available PLSInputs keys
    pls_inputs = structures.PLSInputs(**pls_inputs)
    for key in pls_inputs.keys():
        assert hasattr(pls_inputs, key)
        assert np.all(getattr(pls_inputs, key) == pls_inputs[key])

    # test_split and n_split should be None when set to 0
    assert structures.PLSInputs(n_split=0).n_split is None
    assert structures.PLSInputs(test_split=0).test_split is None

    # confirm n_proc inputs are handled appropriately
    assert structures.PLSInputs(n_proc=1).n_proc == 1
    for n_proc in ["max", -1]:
        assert structures.PLSInputs(n_proc=n_proc).n_proc == mp.cpu_count()
    assert structures.PLSInputs(n_proc=-2).n_proc == mp.cpu_count() - 1

    # check input checking for test_size
    with pytest.raises(ValueError):
        structures.PLSInputs(test_size=1)
    with pytest.raises(ValueError):
        structures.PLSInputs(test_size=-0.5)

    # check that PLSInputs rejects disallowed keys
    assert structures.PLSInputs(notakey=10).get("notakey") is None


@pytest.mark.xfail
def test_PLSResults():
    """Test PLSResults structure functionality."""
    assert False


@pytest.mark.xfail
def test_PLSBootResults():
    """Test PLSBootResults structure functionality."""
    assert False


@pytest.mark.xfail
def test_PLSPermResults():
    """Test PLSPermResults structure functionality."""
    assert False


@pytest.mark.xfail
def test_PLSSplitHalfResults():
    """Test PLSSplitHalfResults structure functionality."""
    assert False


@pytest.mark.xfail
def test_PLSCrossValidationResults():
    """Test PLSCrossValidationResults structure functionality."""
    assert False
