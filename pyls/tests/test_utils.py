# -*- coding: utf-8 -*-

import numpy as np
import pytest
import pyls

rs = np.random.RandomState(1234)


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
    groups = np.hstack([[1] * 10, [2] * 10])

    xcorr = pyls.utils.xcorr(X, Y)
    assert xcorr.shape == (25, 200)
    xcorr = pyls.utils.xcorr(X, Y, groups)
    assert xcorr.shape == (25 * 2, 200)


def test_dummycode():
    groups = np.hstack([[1] * 15,
                        [2] * 15])
    dummy = pyls.utils.dummy_code(groups)
    for n, grp in enumerate(dummy.T.astype('bool'), 1):
        assert np.all(groups[grp] == n)
