# pyls
A python 3 implementation of Partial Least Squares (PLS) cross-covariance decomposition.

[![Build Status](https://travis-ci.org/rmarkello/pyls.svg?branch=master)](https://travis-ci.org/rmarkello/pyls)
[![codecov](https://codecov.io/gh/rmarkello/pyls/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/pyls)

## Requirements
This package requires python >= 3.5, and a [few helpful modules](https://github.com/rmarkello/pyls/blob/master/requirements.txt).

## Installation
Using `git clone` and `python setup.py install` is currently the only means for installation. There are plans to get this set up on PyPI and, perhaps, conda-forge, but those are longer term and would require this to be stable!

## Purpose
**Q**: "Doesn't [`sklearn.cross_decomposition.PLSSVD`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html) do what `pyls.BehavioralPLS` does?"  
**A**: Well, sort of... To quote the documentation, `PLSSVD` "simply perform[s] a svd on the crosscovariance matrix: `X’Y`."

The PLS implementation in the current package mirrors the functionality originally introduced by [McIntosh et al., (1996)](https://www.ncbi.nlm.nih.gov/pubmed/9345485) in their [Matlab toolbox](https://www.rotman-baycrest.on.ca/index.php?section=84). However, while the MATLAB toolbox has a significant amount of functionality dedicated to integrating neuroimaging-specific tools (i.e., loading M/EEG and fMRI data), the current Python package aims to implement and expand on only the core statistical functions.

Thus, this package goes beyond the `sklearn` implementation in a few respects. Namely, `pyls`:
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
                n_perm=100,        # num of permutations for signif. testing
                n_boot=50,         # num of bootstraps for resampling
                n_split=50,        # num of split-half for reliability testing
                seed=1234)         # seed for reproducibility

# behavioral PLS operates on the cross-covariance of X and Y
>>> bpls = pyls.BehavioralPLS(X, Y, **opts)
>>> bpls.results  # the results structure
PLSResults(u, s, v, usc, vsc, boot_result, perm_result, perm_splithalf, inputs, s_varexp)

# mean-centered PLS finds components of X that differentiate between groups/conditions
>>> mpls = pyls.MeanCenteredPLS(X, **opts)
>>> mpls.results
PLSResults(u, s, v, usc, vsc, boot_result, perm_result, perm_splithalf, inputs, s_varexp)
```

The results are currently formatted the same as how the PLS Matlab toolbox would generate them. The results object has a doc-string describing what each output represents, so while we work on [getting some documentation](https://github.com/rmarkello/pyls/issues/19) you can rely on those for some insight!


## How to get involved
We're thrilled to welcome new contributors! If you're interesting in getting involved, you should start by reading our [contributing guidelines](https://github.com/rmarkello/pyls/blob/master/CONTRIBUTING.md) and [code of conduct](https://github.com/rmarkello/pyls/blob/master/Code_of_Conduct.md).

Once you're done with that, you can take a look at our [roadmap](https://github.com/rmarkello/pyls/issues/26) and the accompanying [project](https://github.com/rmarkello/pyls/projects/1). If you find something you'd like to work on, head on over to the relevant [issue](https://github.com/rmarkello/pyls/issues) and let us know.

If you've found a bug, are experiencing a problem, or have a question, create a new issue with some information about it!
