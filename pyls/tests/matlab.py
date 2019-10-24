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


def assert_func_equiv(a, b, corr=0.975, ftol=0.01):
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
    ftol : float, optional
        If len(a) and len(b) <= 2, the correlation cannot be used to assess
        functional equivalence. Instead, this specifies the numerical tolerance
        permitted between corresponding values in the two vectors.

    Raises
    ------
    AssertionError
        If `a` and `b` are not functionally equivalent
    """

    if len(a) == 1 and len(b) == 1:  # can't do anything here, really...
        return
    elif len(a) <= 2 and len(b) <= 2:  # can't correlate length 2 array...
        assert np.allclose(np.sign(a), np.sign(b))
        if ftol is not None:
            assert np.all(np.abs(a - b) < ftol)
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


def compare_python_matlab(python, matlab, *, atol=1e-4, corr=0.975, alpha=0.05,
                          ftol=0.01):
    """
    Compares PLS results generated from `python` and `matlab`

    Due to floating point differences in linear algebra routines like SVD that
    propagate through permutation testing and bootstrap resampling, we cannot
    expected that PLS results from Python and Matlab will generate _exactly_
    the same results. This function compares the numerical eqivalence of
    results we do expect to be exact, and assesses the functional equivalence
    of the remaining results using correlations and alpha testing, as
    appropriate.

    Parameters
    ----------
    python : :obj:`pyls.structures.PLSResults`
        PLSResults object generated from Python
    matlab : :obj:`pyls.structures.PLSResults`
        PLSResults object generated from Matlab
    atol : float, optional
        Absolute tolerance permitted between `python` and `matlab` results
        that should have numerical equivalency. Default: 1e-4
    corr : [0, 1] float, optional
        Minimum correlation expected between `python` and `matlab` results
        that can't be expected to retain numerical equivalency. Default: 0.975
    alpha : [0, 1] float, optional
        Alpha level for assessing significance of latent variables, used to
        compare whether `python` and `matlab` results retain same functional
        significance. Default: 0.05
    ftol : float, optional
        If len(a) and len(b) <= 2, the correlation ( `corr`) cannot be used to
        assess functional equivalence. Instead, this value specifies the
        numerical tolerance allowed between corresponding values in the two
        vectors. Default: 0.01

    Returns
    -------
    equivalent : bool
        Whether PLSResults objects stored in `python` and `matlab` are
        functionally (not necessarily exactly numerically) equivalent
    reason : str
        If `equivalent=False`, reason for failure; otherwise, empty string
    """

    if not isinstance(python, pyls.PLSResults):
        raise ValueError('Provided `python` object must be a pyls.PLSResults '
                         'instance, not {}.'.format(type(python)))
    if not isinstance(matlab, pyls.PLSResults):
        raise ValueError('Provided `matlab` object must be a pyls.PLSResults '
                         'instance, not {}.'.format(type(matlab)))

    # singular values close to 0 cannot be considered because they're random
    keep = ~np.isclose(python['singvals'], 0)

    # check top-level results (only for shared keys)
    for k in python.keys():
        if isinstance(python[k], np.ndarray) and (k in matlab):
            a, b = python[k][..., keep], matlab[k][..., keep]
            try:
                assert_num_equiv(a, b, atol=atol)
            except AssertionError:
                return False, k

    # check pvals for functional equivalence
    if matlab.get('permres', {}).get('pvals') is not None:
        a = python['permres']['pvals'][keep]
        b = matlab['permres']['pvals'][keep]
        try:
            assert_func_equiv(a, b, corr, ftol=ftol)
            assert_pvals_equiv(a, b, alpha)
        except AssertionError:
            return False, 'permres.pvals'

    # check bootstraps for functional equivalence
    if matlab.get('bootres', {}).get('x_weights_normed') is not None:
        a = python['bootres']['x_weights_normed'][..., keep]
        b = matlab['bootres']['x_weights_normed'][..., keep]
        try:
            assert_func_equiv(a, b, corr, ftol=ftol)
        except AssertionError:
            return False, 'bootres.x_weights_normed'

    # check splitcorr for functional equivalence
    if matlab.get('splitres', {}).get('ucorr') is not None:
        a, b = python['splitres'], matlab['splitres']
        try:
            for k in ['ucorr', 'vcorr']:
                assert_func_equiv(a[k][keep], b[k][keep], corr, ftol=ftol)
        except AssertionError:
            return False, 'splitres.{}'.format(k)

    return True, ''


def assert_matlab_equivalence(fname, method=None, *, atol=1e-4, corr=0.975,
                              alpha=0.05, ftol=0.01, **kwargs):
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
    atol : float, optional
        Absolute tolerance permitted between `python` and `matlab` results
        that should have numerical equivalency. Default: 1e-4
    corr : [0, 1] float, optional
        Minimum correlation expected between `python` and `matlab` results
        that can't be expected to retain numerical equivalency. Default: 0.975
    alpha : [0, 1] float, optional
        Alpha level for assessing significance of latent variables, used to
        compare whether `python` and `matlab` results retain same functional
        significance. Default: 0.05
    ftol : float, optional
        If len(a) and len(b) <= 2, the correlation ( `corr`) cannot be used to
        assess functional equivalence. Instead, this value specifies the
        numerical tolerance allowed between corresponding values in the two
        vectors. Default: 0.01
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
    if 'n_split' not in matlab['inputs']:
        matlab['inputs']['n_split'] = None

    # get PLS method
    fcn = None
    if method is None:
        if matlab['inputs']['method'] == 1:
            fcn = pyls.meancentered_pls
        elif matlab['inputs']['method'] == 3:
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
    matlab['inputs']['seed'] = 1234
    matlab['inputs']['verbose'] = False
    # don't update n_split if it was previously set to None
    if matlab['inputs']['n_split'] is None:
        if 'n_split' in kwargs:
            kwargs.pop('n_split')
    matlab['inputs'].update(kwargs)

    # run PLS
    python = fcn(**matlab['inputs'])
    equiv, reason = compare_python_matlab(python, matlab, atol=atol, corr=corr,
                                          alpha=alpha, ftol=ftol)

    if not equiv:
        raise AssertionError('compare_matlab_result failed: {}'.format(reason))
