# -*- coding: utf-8 -*-

import numpy as np
import pyls

rs = np.random.RandomState(1234)


def test_RefDict():
    class BlahDict(pyls.utils.ResDict):
        allowed = ['test']

    d = pyls.utils.ResDict()
    assert str(d) == 'ResDict()'
    assert(str(BlahDict(test={})) == 'BlahDict()')
    assert(str(BlahDict(test=10)) == 'BlahDict(test)')


def test_dummycode():
    groups = [10, 12, 11]
    dummy = pyls.utils.dummy_code(groups)
    assert dummy.shape == (np.sum(groups), len(groups))

    for n, grp in enumerate(dummy.T.astype(bool)):
        assert grp.sum() == groups[n]

    dummy_cond = pyls.utils.dummy_code(groups, n_cond=3)
    assert dummy_cond.shape == (np.sum(groups) * 3, len(groups) * 3)
