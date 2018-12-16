# -*- coding: utf-8 -*-

import numpy as np
import pyls


def assert_num_equiv(a, b, atol=1e-5, drop_last=True):
    """
    Asserts numerical equivalence of `a` and `b`

    Compares numerical equivalence of `a` and `b`, accounting for potential
    sign flips. Uses :func:`numpy.allclose` for assessing equivalence once
    sign flips have been considered.

    Parameters
    ----------
    a, b : array_like
        Arrays to compare for numerical equivalence
    atol : float, optional
        Absolute tolerance for differences in `a` and `b`. Default: 1e-5
    drop_last : bool, optional
        Whether to consider the last feature of `a` and `b`. Default: True

    Raises
    ------
    AssertionError
        If `a` and `b` are not numerically equivalent to `atol`
    """

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
    """
    Asserts "functional" equivalence of `a` and `b`

    Given the numerical instabilities of SVD between Matlab and Python we
    cannot always assume numerical equivalence, especially when permutation
    testing and bootstrap resampling are considered. This function thus
    considers whether results are "functionally" equivalent, where functional
    equivalence is defined by the correlation of `a` and `b` (if both are one-
    dimensional) or the correlation of columns of `a` and `b` (if both are two-
    dimensional). Correlations must surpass provided `corr` to be considered
    functionally equivalent.

    Parameters
    ----------
    a, b : array_like
        Arrays to compare for functional equivalence
    corr : [0, 1] float, optional
        Correlation that must be surpassed in order to achieve functional
        equivalence between `a` and `b`. Default: 0.99
    drop_last : bool, optional
        Whether to consider the last feature of `a` and `b`. Default: True

    Raises
    ------
    AssertionError
        If `a` and `b` are not functionally equivalent
    """

    # can't perform correlation on length 2 array...
    if len(a) <= 2 and len(b) <= 2:
        if drop_last:  # only one measurement, can't do anything, just return
            return
        # ensure that the sign change is consistent between arrays
        diff = a - b
        assert np.all(np.sign(diff) == 1) or np.all(np.sign(diff) == -1)
        return

    if a.ndim > 1:
        corrs = pyls.compute.efficient_corr(a, b)
        if drop_last:
            corrs = corrs[:-1]
    else:
        if drop_last:
            a, b = a[:-1], b[:-1]
        corrs = np.corrcoef(a, b)[0, 1]
    assert np.all(np.abs(np.around(corrs, 2)) >= corr)


def assert_pvals_equiv(a, b, alpha=0.05, drop_last=True):
    """
    Asserts that p-values in `a` and `b` achieve same statistical significance

    Uses `alpha` to determine significance threshold and ensures that
    corresponding p-values in `a` and `b` both reject or fail to reject the
    null hypothesis.

    Parameters
    ----------
    a, b : array_like
        Arrays of p-values to be considered
    alpha : [0, 1] float, optional
        Alpha to set statistical significance threshold. Default: 0.05
    drop_last : bool, optional
        Whether to consider the last feature of `a` and `b`. Default: True

    Raises
    ------
    AssertionError
        If p-values in `a` and `b` do not achieve identical statistical
        significance thresholds
    """

    if a.shape != b.shape:
        assert False
    if drop_last:
        a, b = a[:-1], b[:-1]
    assert np.all((a < alpha) == (b < alpha))


def compare_python_matlab(python, matlab, method, corr=0.99, alpha=0.05):
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
        Alpha level for assessing significance of latent variables, used to
        compare whether `python` and `matlab` results retain same functional
        significance

    Returns
    ------
    equivalent : bool
        Whether PLSResults objects stored in `python` and `matlab` are
        functionally (not necessarily exactly numerically) equivalent
    reason : str
        If `equivalent=False`, reason for failure; otherwise, empty string
    """

    drop_last = method == 'meancentered'

    # check top-level results attributes for numerical equivalence
    # only do this for singular values that are > 0
    keep = ~np.isclose(python.s, 0)
    for k in python.keys():
        if isinstance(k, np.ndarray):
            try:
                assert_num_equiv(python[k][:, keep], matlab[k][:, keep],
                                 drop_last=drop_last)
            except AssertionError:
                return False, k

    # check pvals for functional equivalence
    if matlab.get('permres', {}).get('pvals') is not None:
        try:
            assert_func_equiv(python.permres.pvals, matlab.permres.pvals,
                              corr, drop_last=drop_last)
            assert_pvals_equiv(python.permres.pvals, matlab.permres.pvals,
                               alpha, drop_last=drop_last)
        except AssertionError:
            return False, 'permres.pvals'

    # check bootstraps for functional equivalence
    if matlab.get('bootres', {}).get('bootstrapratios') is not None:
        try:
            assert_func_equiv(python.bootres.bootstrapratios[:, keep],
                              matlab.bootres.bootstrapratios[:, keep],
                              corr, drop_last=drop_last)
        except AssertionError:
            return False, 'bootres.bootstrapratios'

    # check splitcorr for functional equivalence
    if matlab.get('splitres', {}).get('ucorr') is not None:
        # lenient functional equivalence
        try:
            for k in ['ucorr', 'vcorr']:
                assert_func_equiv(python.splitres[k][keep],
                                  matlab.splitres[k][keep], corr,
                                  drop_last=drop_last)
            # only consider the splithalf pvalues of the permuted LVs that are
            # significant as these are the only ones that we would consider,
            # functionally speaking
            for k in ['ucorr_pvals', 'vcorr_pvals']:
                pk = python.permres.pvals < alpha
                assert_pvals_equiv(python.splitres[k][pk],
                                   matlab.splitres[k][pk],
                                   alpha, drop_last=drop_last)
        except AssertionError:
            return False, 'splitres.{}'.format(k)

    return True, ''


def assert_matlab_equivalence(fname, method=None, corr=0.99, alpha=0.05,
                              **kwargs):
    """
    Compares Matlab PLS results stored in `fname` with Python-generated results

    Loads `fname` using :func:`pyls.import_matlab_result`, re-runs analysis,
    and then compares results using :func:`pyls.tests.compare_matlab_result`.

    Parameters
    ----------
    fname : str
        Path to Matlab PLS results
    method : function, optional
        PLS function to use to re-run analysis from `fname`. If not specified
        will try and determine method from `fname`. Default: None
    corr : [0, 1] float, optional
        Minimum correlation expected between Matlab and Python PLS results that
        can't be expected to retain numerical equivalency (i.e., permutation
        results, bootstrapping results)
    alpha : [0, 1] float, optional
        Alpha level for assessing significance of latent variables, used to
        compare whether Matlab and Python PLS results retain same functional
        significance
    kwargs : optional
        Key-value arguments to provide to PLS analysis. May override arguments
        specified in `fname`

    Raises
    ------
    AssertionError
        If PLS results generated by Python are not the same as those stored in
        `fname`
    """
    # load matlab result
    matlab = pyls.matlab.import_matlab_result(fname)

    # fix n_split default (if not specified in matlab assume 0)
    if not hasattr(matlab.inputs, 'n_split'):
        matlab.inputs.n_split = 0

    # run PLS
    if method is None:
        if matlab.inputs.method == 1:
            fcn = pyls.meancentered_pls
        elif matlab.inputs.method == 3:
            fcn = pyls.behavioral_pls
        else:
            raise ValueError('Cannot determine PLS method used to generate {}'
                             'from file. Please provide `method` function '
                             'to make_matlab_comparison() call.'.format(fname))
    else:
        fcn = method

    # use seed for reproducibility of re-analysis
    matlab.inputs.seed = 1234
    matlab.inputs.verbose = False
    matlab.inputs.update(kwargs)

    python = fcn(**matlab.inputs)
    method = ['behavioral', 'meancentered'][fcn == pyls.meancentered_pls]
    equiv, reason = compare_python_matlab(python, matlab, method, corr, alpha)

    if not equiv:
        raise AssertionError('compare_matlab_result failed: {}'.format(reason))
