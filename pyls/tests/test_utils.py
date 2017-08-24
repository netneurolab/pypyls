#!/usr/bin/env python

import pytest
import os.path as op
import numpy as np
import pyls


def test_zscore():
    pyls.utils.zscore([[1]]*10)
    pyls.utils.zscore(np.ones((10,1)))


def test_normalize():
    X = np.random.rand(10,10)
    pyls.utils.normalize(X,dim=0)
    pyls.utils.normalize(X,dim=1)


def test_flatten_niis():
    fn = op.join(op.dirname(op.abspath(__file__)),'data','blank.nii.gz')
    pyls.utils.flatten_niis([fn,fn])
    with pytest.raises(ValueError): pyls.utils.flatten_niis([fn,fn],thresh=-1)
