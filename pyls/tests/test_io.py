# -*- coding: utf-8 -*-

import os.path as op
import h5py
import pytest
import pyls


def test_load_save(testdir, mpls_results, bpls_results):
    for res, fn in zip([mpls_results, bpls_results], ['mpls', 'bpls']):
        fname = pyls.save_results(op.join(testdir, fn), res)
        assert op.isfile(fname)
        assert h5py.is_hdf5(fname)

    mpls = pyls.load_results(op.join(testdir, 'mpls.hdf5'))
    bpls = pyls.load_results(op.join(testdir, 'bpls.hdf5'))

    assert mpls == mpls_results and mpls != bpls_results
    assert bpls == bpls_results and bpls != mpls_results

    with pytest.raises(TypeError):
        pyls.load_results(testdir)
