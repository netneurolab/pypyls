"""Tests for the I/O functions."""

import os.path as op
import h5py
import pytest
import pyls


def test_load_save(testdir, mpls_results, bpls_results):
    """Test saving and loading of results."""
    for res, fn in zip([mpls_results, bpls_results], ["mpls", "bpls"]):
        fname = pyls.save_results(op.join(testdir, fn), res)
        assert op.isfile(fname)
        assert h5py.is_hdf5(fname)
        assert pyls.load_results(fname) == res

    with pytest.raises(TypeError):
        pyls.load_results(testdir)
