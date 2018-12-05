# -*- coding: utf-8 -*-

import numpy as np
import pytest
import pyls


@pytest.fixture(scope='session')
def testdir(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp('data')
    return data_dir


@pytest.fixture(scope='session')
def mpls_results():
    Xf = 1000
    subj = 100
    rs = np.random.RandomState(1234)
    return pyls.meancentered_pls(rs.rand(subj, Xf), n_cond=2,
                                 n_perm=10, n_boot=10, n_split=10)


@pytest.fixture(scope='session')
def bpls_results():
    Xf = 1000
    Yf = 100
    subj = 100
    rs = np.random.RandomState(1234)
    return pyls.behavioral_pls(rs.rand(subj, Xf), rs.rand(subj, Yf),
                               n_perm=10, n_boot=10, n_split=10)
