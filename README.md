# pyls
A python 3 implementation of Partial Least Squares (PLS) cross-covariance decomposition.

[![Build Status](https://travis-ci.org/rmarkello/pyls.svg?branch=master)](https://travis-ci.org/rmarkello/pyls)
[![codecov](https://codecov.io/gh/rmarkello/pyls/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/pyls)

## Requirements
This package requires python >= 3.5, and a [few helpful modules](https://github.com/rmarkello/pyls/blob/master/requirements.txt).

## Installation
Using `git clone` and `python setup.py install` is currently the only means for installation. There are plans to get this set up on PyPI and, perhaps, conda-forge, but those are longer term and would require this to be stable!

## Purpose
**Q**: "[`sklearn.cross_decomposition.PLSSVD`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html) implements this, doesn't it?"  
**A**: Well, sort of... To quote the documentation of `PLSSVD`: "Simply perform a svd on the crosscovariance matrix: `X’Y`." This package does a bit more than that:

The PLS implementation in the current package mirrors the functionality originally introduced by [McIntosh et al., (1996)](https://www.ncbi.nlm.nih.gov/pubmed/9345485); indeed, this is in many respects a translation of the [PLS Matlab toolbox](https://www.rotman-baycrest.on.ca/index.php?section=84). However, while that toolbox has a significant amount of functionality dedicated to integrating neuroimaging-specific tools (i.e., loading M/EEG and fMRI data), the current package aims to only implement the statistical functions.

Thus, this package's functionality expands on the `sklearn` implementation in a few respects. Namely, `pyls`:
1. Has integrated significance and reliability testing via built-in permutation testing and bootstrap resampling, and
2. Implements [mean-centered PLS](https://www.ncbi.nlm.nih.gov/pubmed/20656037) for multivariate group/condition comparisons.

## Usage
Currently both `BehavioralPLS` and `MeanCenteredPLS` are implemented. A brief example of their usage:

```python
>>> import pyls

>>> X = np.random.rand(20, 10000)  # high dimensional data (e.g., neural)
>>> Y = np.random.rand(20, 10)     # lower dimensional data (e.g., behavioral)
>>> opts = dict(groups=[5, 5],     # 2 groups of 5 subjects each
                n_cond=2,          # 2 conditions / group
                n_perm=100,        # for permutation testing
                n_boot=50,         # for bootstrap resampling
                n_split=50,        # for split-half resampling
                seed=1234)         # for reproducibility

# a behavioral PLS operates on the cross-covariance of X and Y
>>> bpls = pyls.BehavioralPLS(X, Y, **opts)
>>> bpls.results  # get the results structure
PLSResults(u, s, v, usc, vsc, boot_result, perm_result, perm_splithalf, inputs, s_varexp)

# a mean-centered PLS tries to differentiate X amongst the groups/conditions
>>> mpls = pyls.MeanCenteredPLS(X, **opts)
>>> mpls.results
PLSResults(u, s, v, usc, vsc, boot_result, perm_result, perm_splithalf, inputs, s_varexp)
```
