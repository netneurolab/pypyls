# -*- coding: utf-8 -*-

import os.path as op
import pkg_resources
import pytest
import pyls

data_dir = pkg_resources.resource_filename('pyls', 'tests/data')
EXAMPLES = ['mpls_multigroup_onecond_nosplit.mat',
            'mpls_multigroup_onecond_split.mat',
            'bpls_onegroup_onecond_nosplit.mat',
            'bpls_onegroup_onecond_split.mat',
            'resultonly.mat']

attrs = [
    'x_weights', 'singvals', 'y_weights', 'x_scores', 'permres', 'bootres',
    'inputs'
]


@pytest.mark.parametrize('fname', EXAMPLES)
def test_import_matlab(fname):
    res = pyls.matlab.import_matlab_result(op.join(data_dir, fname))
    # make sure the mat file cast appropriately
    assert isinstance(res, pyls.structures.PLSResults)
    # make sure all the attributes are there (don't check outputs)
    for attr in attrs:
        assert hasattr(res, attr)
    if '_split' in fname:
        assert hasattr(res, 'splitres')


def test_errors():
    with pytest.raises(ValueError):
        pyls.matlab.import_matlab_result(op.join(data_dir, 'empty.mat'))
