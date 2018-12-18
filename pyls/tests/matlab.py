# -*- coding: utf-8 -*-

import numpy as np
import pyls


def assert_num_equiv(a, b, atol=1e-4):
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
        Absolute tolerance for differences in `a` and `b`. Default: 1e-4

    Raises
    ------
    AssertionError
        If `a` and `b` are not numerically equivalent to `atol`
    """

    # signs may be flipped so adjust accordingly
    flip = 1 * np.all(np.sign(b / a) == 1, axis=0, keepdims=True)
    flip[flip == 0] = -1
    diff = a - (b * flip)

    assert np.allclose(diff, 0, atol=atol)


def assert_func_equiv(a, b, corr=0.975):
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

    Raises
    ------
    AssertionError
        If `a` and `b` are not functionally equivalent
    """

    if len(a) == 1 and len(b) == 1:  # can't do anything here, really...
        return
    elif len(a) <= 2 and len(b) <= 2:  # can't correlate length 2 array...
        # ensure that the sign change is consistent between arrays
        diff = a - b
        assert np.all(np.sign(diff) == 1) or np.all(np.sign(diff) == -1)
        return

    if a.ndim > 1:
        corrs = pyls.compute.efficient_corr(a, b)
    else:
        corrs = np.corrcoef(a, b)[0, 1]

    assert np.all(np.abs(corrs) >= corr)


def assert_pvals_equiv(a, b, alpha=0.05):
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

    Raises
    ------
    AssertionError
        If p-values in `a` and `b` do not achieve identical statistical
        significance thresholds
    """

    assert np.all((a < alpha) == (b < alpha))


def compare_python_matlab(python, matlab, corr=0.975, alpha=0.05):
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
    corr : [0, 1] float, optional
        Minimum correlation expected between `python` and `matlab` results
        that can't be expected to retain numerical equivalency
    alpha : [0, 1] float, optional
        Alpha level for assessing significance of latent variables, used to
        compare whether `python` and `matlab` results retain same functional
        significance

    Returns
    -------
    equivalent : bool
        Whether PLSResults objects stored in `python` and `matlab` are
        functionally (not necessarily exactly numerically) equivalent
    reason : str
        If `equivalent=False`, reason for failure; otherwise, empty string
    """

    # singular values close to 0 cannot be considered because they're random
    keep = ~np.isclose(python.s, 0)

    # check top-level results
    for k in python.keys():
        if isinstance(python[k], np.ndarray):
            try:
                assert_num_equiv(python[k][..., keep], matlab[k][..., keep])
            except AssertionError:
                return False, k

    # check pvals for functional equivalence
    if matlab.get('permres', {}).get('pvals') is not None:
        try:
            assert_func_equiv(python.permres.pvals[keep],
                              matlab.permres.pvals[keep],
                              corr)
            assert_pvals_equiv(python.permres.pvals[keep],
                               matlab.permres.pvals[keep],
                               alpha)
        except AssertionError:
            return False, 'permres.pvals'

    # check bootstraps for functional equivalence
    if matlab.get('bootres', {}).get('bootstrapratios') is not None:
        try:
            assert_func_equiv(python.bootres.bootstrapratios[..., keep],
                              matlab.bootres.bootstrapratios[..., keep],
                              corr)
        except AssertionError:
            return False, 'bootres.bootstrapratios'

    # check splitcorr for functional equivalence
    if matlab.get('splitres', {}).get('ucorr') is not None:
        # lenient functional equivalence
        try:
            for k in ['ucorr', 'vcorr']:
                assert_func_equiv(python.splitres[k][keep],
                                  matlab.splitres[k][keep], corr)
        except AssertionError:
            return False, 'splitres.{}'.format(k)

    return True, ''


def assert_matlab_equivalence(fname, method=None, corr=0.975, alpha=0.05,
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

    # get PLS method
    fcn = None
    if method is None:
        if matlab.inputs.method == 1:
            fcn = pyls.meancentered_pls
        elif matlab.inputs.method == 3:
            fcn = pyls.behavioral_pls
    elif isinstance(method, str):
        if method == 'meancentered':
            fcn = pyls.meancentered_pls
        elif method == 'behavioral':
            fcn = pyls.behavioral_pls
    elif callable(method):
        if method in [pyls.meancentered_pls, pyls.behavioral_pls]:
            fcn = method

    if fcn is None:
        raise ValueError('Cannot determine PLS method used to generate {}'
                         'from file. Please provide `method` argument.'
                         .format(fname))

    # use seed for reproducibility of re-analysis
    matlab.inputs.seed = 1234
    matlab.inputs.verbose = False
    matlab.inputs.update(kwargs)

    # run PLS
    python = fcn(**matlab.inputs)
    equiv, reason = compare_python_matlab(python, matlab, corr, alpha)

    if not equiv:
        raise AssertionError('compare_matlab_result failed: {}'.format(reason))
