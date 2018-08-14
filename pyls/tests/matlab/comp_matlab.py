# -*- coding: utf-8 -*-

from pkg_resources import resource_filename
import numpy as np
import pyls

EXAMPLES = [
    'mpls_multigroup_onecond_nosplit.mat',
    'mpls_multigroup_onecond_split.mat',
    'bpls_onegroup_onecond_nosplit.mat',
    'bpls_onegroup_onecond_split.mat'
]


def assert_num_equiv(python, matlab, atol=1e-4):
    # signs may be flipped so just take difference of absolute values
    diff = np.abs(python) - np.abs(matlab)

    # the last LV is always screwed up so ignore it
    if diff.ndim > 1:
        diff = diff[:, :-1]
    else:
        diff = diff[:-1]

    assert np.allclose(diff, 0, atol=atol)


def assert_func_equiv(a, b, corr=0.99):
    if a.ndim > 1:
        corrs = np.array([np.corrcoef(a[:, i], b[:, i])[0, 1] for i in
                          range(a.shape[-1] - 1)])
    else:
        corrs = np.corrcoef(a[:-1], b[:-1])[0, 1]
    assert np.all(np.abs(corrs) > corr)


def assert_pvals_equiv(a, b, alpha=0.05):
    assert np.all((a[:-1] < alpha) == (b[:-1] < alpha))


def make_comparison(fname, corr=0.99, alpha=0.05):
    # load matlab result
    fname = resource_filename('pyls', 'tests/data/{}'.format(fname))
    matlab = pyls.matlab.import_matlab_result(fname)

    # fix n_split default (if not specified in matlab assume 0)
    if not hasattr(matlab.inputs, 'n_split'):
        matlab.inputs.n_split = 0

    # run PLS
    fcn = pyls.behavioral_pls if 'bpls' in fname else pyls.meancentered_pls
    python = fcn(**matlab.inputs, seed=1234)

    # check top-level results attributes for numerical equivalence
    assert np.all([assert_num_equiv(python[k], matlab[k])[0]
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
        # lenient functional equivalence
        for k in ['ucorr', 'vcorr']:
            assert assert_num_equiv(python.splitres[k], matlab.splitres[k],
                                    atol=0.1)
        for k in ['ucorr_pvals', 'vcorr_pvals']:
            assert_pvals_equiv(python.splitres[k], matlab.splitres[k], alpha)
