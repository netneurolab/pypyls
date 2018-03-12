# -*- coding: utf-8 -*-

import os.path as op
import pkg_resources
import pytest
import pyls

data_dir = pkg_resources.resource_filename('pyls', 'tests/data')
EXAMPLES = ['mpls_multigroup_onecond_nosplit.mat',
            'mpls_multigroup_onecond_split.mat',
            'bpls_onegroup_onecond_nosplit.mat',
            'bpls_onegroup_onecond_split.mat']


def test_import_matlab():
    for fname in EXAMPLES:
        res = pyls.matlab.import_matlab_result(op.join(data_dir, fname))
        assert isinstance(res, pyls.base.PLSResults)
