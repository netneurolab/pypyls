"""Tests for the compute module."""

import numpy as np
import pytest
import pyls

rs = np.random.RandomState(1234)


def test_normalize():
    """Test the normalize function for axis-wise normalization."""
    X = rs.rand(10, 10)
    out = pyls.compute.normalize(X, axis=0)
    assert np.allclose(np.sum(out**2, axis=0), 1)

    out = pyls.compute.normalize(X, axis=1)
    assert np.allclose(np.sum(out**2, axis=1), 1)


def test_xcorr():
    """Test the xcorr function for cross-correlation computation."""
    X = rs.rand(20, 200)
    Y = rs.rand(20, 25)

    xcorr = pyls.compute.xcorr(X, Y)
    assert xcorr.shape == (25, 200)
    xcorr = pyls.compute.xcorr(X, Y, norm=True)
    assert xcorr.shape == (25, 200)

    with pytest.raises(ValueError):
        pyls.compute.xcorr(X[:, 0], Y)
    with pytest.raises(ValueError):
        pyls.compute.xcorr(X[:, 0], Y[:, 0])
    with pytest.raises(ValueError):
        pyls.compute.xcorr(X[0:10], Y)


def test_efficient_corr():
    """Test the efficient_corr function for optimized correlation computation."""
    x, y = rs.rand(100), rs.rand(100, 10)
    assert pyls.compute.efficient_corr(x, y).shape == (10,)
    x = rs.rand(100, 10)
    assert pyls.compute.efficient_corr(x, y).shape == (10,)

    x = rs.rand(100, 2)
    with pytest.raises(ValueError):
        pyls.compute.efficient_corr(x, y)

    x, y = np.ones((100, 2)), np.ones((100, 2)) * 5
    x[50:, 0], y[50:, 0] = 2, 6
    x[50:, 1], y[50:, 1] = 2, 4
    assert np.allclose(pyls.compute.efficient_corr(x, y), np.array([1.0, -1.0]))
