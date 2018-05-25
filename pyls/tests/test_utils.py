# -*- coding: utf-8 -*-

import numpy as np
import pytest
import pyls

rs = np.random.RandomState(1234)


def test_DefDict():
    d = pyls.utils.DefDict()
    print(d)


def test_zscore():
    out = pyls.utils.zscore([[1]] * 10)
    assert np.allclose(out, 0)

    out = pyls.utils.zscore(rs.rand(10, 10))
    assert out.shape == (10, 10)
    assert not np.allclose(out, 0)


def test_normalize():
    X = rs.rand(10, 10)
    out = pyls.utils.normalize(X, axis=0)
    assert np.allclose(np.sum(out**2, axis=0), 1)

    out = pyls.utils.normalize(X, axis=1)
    assert np.allclose(np.sum(out**2, axis=1), 1)


def test_xcorr():
    X = rs.rand(20, 200)
    Y = rs.rand(20, 25)

    xcorr = pyls.utils.xcorr(X, Y)
    assert xcorr.shape == (25, 200)
    xcorr = pyls.utils.xcorr(X, Y, norm=False)
    assert xcorr.shape == (25, 200)

    with pytest.raises(ValueError):
        pyls.utils.xcorr(X[:, 0], Y)
    with pytest.raises(ValueError):
        pyls.utils.xcorr(X[:, 0], Y[:, 0])
    with pytest.raises(ValueError):
        pyls.utils.xcorr(X[0:10], Y)


def test_dummycode():
    groups = [10, 12, 11]
    dummy = pyls.utils.dummy_code(groups)
    assert dummy.shape == (np.sum(groups), len(groups))

    for n, grp in enumerate(dummy.T.astype(bool)):
        assert grp.sum() == groups[n]

    dummy_cond = pyls.utils.dummy_code(groups, n_cond=3)
    assert dummy_cond.shape == (np.sum(groups) * 3, len(groups) * 3)
