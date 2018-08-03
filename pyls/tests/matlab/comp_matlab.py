# -*- coding: utf-8 -*-

import os.path as op
import pkg_resources
import numpy as np
import pytest
import pyls

data_dir = pkg_resources.resource_filename('pyls', 'tests/data')
EXAMPLES = ['mpls_multigroup_onecond_nosplit.mat',
            'mpls_multigroup_onecond_split.mat',
            'bpls_onegroup_onecond_nosplit.mat',
            'bpls_onegroup_onecond_split.mat',
            'resultonly.mat']


def assert_func_equiv(a, b, corr=0.99):
    if a.ndim > 1:
        corrs = np.array([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in
                          range(a.shape[-1])])
    else:
        corrs = np.corrcoef(a, b)[0, 1]
    assert np.all(np.abs(corrs) > corr)


def assert_pvals_equiv(a, b, alpha=0.05):
    assert np.all((a < alpha) == (b < alpha))


def make_comparison(fname, corr=0.99, alpha=0.05):
    matlab = pyls.matlab.import_matlab_result(op.join(data_dir, fname))
    fcn = pyls.BehavioralPLS if 'bpls' in fname else pyls.MeanCenteredPLS
    python = fcn(**matlab.inputs)

    # check top-level results attributes for numerical equivalence
    assert np.all([pyls.matlab.comp_python_matlab(python[k], matlab[k])[0]
                   for k in python.keys() if isinstance(k, np.ndarray)])

    # check pvals for functional equivalence
    if matlab.get('permres', {}).get('pvals') is not None:
        assert_func_equiv(python.permres.pvals, matlab.permres.pvals, corr)
        assert_pvals_equiv(python.permres.pvals, matlab.permres.pvals, alpha)

    # check bootstraps for functional equivalence
    if matlab.get('bootres', {}).get('bootstrapratios') is not None:
        assert_func_equiv(python.bootres.bootstrapratios,
                          matlab.bootres.bootstrapratios,
                          corr)

    # check splitcorr for functional equivalence
    if matlab.get('splitres', {}).get('ucorr') is not None:
        for k in ['ucorr', 'vcorr']:
            assert_func_equiv(python.splitres[k], matlab.splitres[k], corr)
        for k in ['ucorr_pvals', 'vcorr_pvals']:
            assert_pvals_equiv(python.splitres[k], matlab.splitres[k], alpha)
