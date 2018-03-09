# -*- coding: utf-8 -*-

import os.path as op
import pkg_resources
import pytest
import pyls

data_dir = pkg_resources.resource_filename('pyls', 'tests/data')
fnames = ['meancenteredpls.mat',
          'meancenteredpls_splithalf.mat',
          'behavioralpls.mat',
          'behavioralpls_splithalf.mat']


def test_import_matlab():
    for fname in fnames:
        res = pyls.matlab.import_matlab_result(op.join(data_dir, fname))
        assert isinstance(res, pyls.base.PLSResults)
