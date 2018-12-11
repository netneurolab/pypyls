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


def assert_num_equiv(a, b, atol=1e-5, drop_last=True):
    # signs may be flipped so adjust accordingly
    flip = 1 * np.all(np.sign(b / a) == 1, axis=0, keepdims=True)
    flip[flip == 0] = -1
    diff = a - (b * flip)

    # the last LV is always screwed up so ignore it
    if drop_last:
        if diff.ndim > 1:
            diff = diff[:, :-1]
        else:
            diff = diff[:-1]

    assert np.allclose(diff, 0, atol=atol)


def assert_func_equiv(a, b, corr=0.99, drop_last=True):
    if a.ndim > 1:
        corrs = pyls.compute.efficient_corr(a, b)
        if drop_last:
            corrs = corrs[:-1]
    else:
        if drop_last:
            a, b = a[:-1], b[:-1]
        corrs = np.corrcoef(a, b)[0, 1]
    assert np.all(np.abs(corrs) > corr)


def assert_pvals_equiv(a, b, alpha=0.05, drop_last=True):
    if drop_last:
        a, b = a[:-1], b[:-1]
    assert np.all((a < alpha) == (b < alpha))


def compare_matlab_result(python, matlab, method, corr=0.99, alpha=0.05):
    """
    Compares PLS results generated from `python` and `matlab`

    Due to floating point differences in linear algebra routines like SVD that
    propagate through permutation testing and bootstrap resampling, we cannot
    expected that PLS results from Python and Matlab will generate _exactly_
    the same results. This function compares the numerical eqivalence of
    results we do expect to be exactly, and assess the functional equivalence
    of the remaining results using correlations and alpha testing, as
    appropriate.

    Parameters
    ----------
    python : :obj:`pyls.PLSResults`
        PLSResults object generated from Python
    matlab : :obj:`pyls.PLSResults`
        PLSResults object generated from Matlab
    method : {'behavioral', 'meancentered'}
        Type of PLS used to generate `python` and `matlab` results
    corr : [0, 1] float, optional
        Minimum correlation expected between `python` and `matlab` results
        that can't be expected to retain numerical equivalency
    alpha : [0, 1] float, optional
        Alpha level for assessing significant of latent variables, used to
        compare whether `python` and `matlab` results retain same functional
        significance

    Returns
    ------
    equivalent : bool
        Whether PLSResults objects stored in `python` and `matlab` are
        functionally (not necessarily exactly numerically) equivalent
    """

    drop_last = method == 'meancentered'

    try:
        # check top-level results attributes for numerical equivalence
        for k in python.keys():
            if isinstance(k, np.ndarray):
                assert_num_equiv(python[k], matlab[k], drop_last=drop_last)

        # check pvals for functional equivalence
        if matlab.get('permres', {}).get('pvals') is not None:
            assert_func_equiv(python.permres.pvals, matlab.permres.pvals,
                              corr, drop_last=drop_last)
            assert_pvals_equiv(python.permres.pvals, matlab.permres.pvals,
                               alpha, drop_last=drop_last)

        # check bootstraps for functional equivalence
        if matlab.get('bootres', {}).get('bootstrapratios') is not None:
            assert_func_equiv(python.bootres.bootstrapratios,
                              matlab.bootres.bootstrapratios,
                              corr, drop_last=drop_last)

        # check splitcorr for functional equivalence
        if matlab.get('splitres', {}).get('ucorr') is not None:
            # lenient functional equivalence
            for k in ['ucorr', 'vcorr']:
                assert_func_equiv(python.splitres[k], matlab.splitres[k], corr,
                                  drop_last=drop_last)
            # only consider the splithalf pvalues of the permuted LVs that are
            # significant as these are the only ones that we would consider,
            # functionally speaking
            for k in ['ucorr_pvals', 'vcorr_pvals']:
                pk = python.permres.pvals < alpha
                mk = matlab.permres.pvals < alpha
                assert_pvals_equiv(python.splitres[k][pk],
                                   matlab.splitres[k][mk],
                                   alpha, drop_last=False)
    except AssertionError:
        return False

    return True


def make_matlab_comparison(fname, method=None):
    # load matlab result
    matlab = pyls.matlab.import_matlab_result(fname)

    # fix n_split default (if not specified in matlab assume 0)
    if not hasattr(matlab.inputs, 'n_split'):
        matlab.inputs.n_split = 0

    # run PLS
    if method is None:
        fcn = pyls.behavioral_pls if 'bpls' in fname else pyls.meancentered_pls
    else:
        fcn = method
    python = fcn(**matlab.inputs, seed=1234)

    method = ['behavioral', 'meancentered']['mpls' in fname]
    assert compare_matlab_result(python, matlab, method)


def test_matlab_comparison():
    for fname in EXAMPLES:
        fname = resource_filename('pyls', 'tests/data/{}'.format(fname))
        make_matlab_comparison(fname)
