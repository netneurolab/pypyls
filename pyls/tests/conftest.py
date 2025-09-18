"""Configuration and fixtures for PLS tests."""

import numpy as np
import pytest
import pyls


@pytest.fixture(scope="session")
def testdir(tmpdir_factory):
    """Create a temporary directory for test data."""
    data_dir = tmpdir_factory.mktemp("data")
    return str(data_dir)


@pytest.fixture(scope="session")
def mpls_results():
    """Generate mean-centered PLS results for testing."""
    Xf = 1000
    subj = 100
    rs = np.random.RandomState(1234)
    return pyls.meancentered_pls(
        rs.rand(subj, Xf), n_cond=2, n_perm=10, n_boot=10, n_split=10
    )


@pytest.fixture(scope="session")
def bpls_results():
    """Generate behavioral PLS results for testing."""
    Xf = 1000
    Yf = 100
    subj = 100
    rs = np.random.RandomState(1234)
    return pyls.behavioral_pls(
        rs.rand(subj, Xf), rs.rand(subj, Yf), n_perm=10, n_boot=10, n_split=10
    )


@pytest.fixture(scope="session")
def pls_inputs():
    """Generate standard PLS input parameters for testing."""
    return dict(
        X=np.random.rand(100, 1000),
        Y=np.random.rand(100, 100),
        groups=[50, 50],
        n_cond=1,
        mean_centering=0,
        n_perm=10,
        n_boot=10,
        n_split=5,
        test_size=0.25,
        test_split=100,
        rotate=True,
        ci=95,
        seed=1234,
        verbose=True,
        permsamples=10,
        bootsamples=10,
    )
