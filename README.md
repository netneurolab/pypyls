# pyls
A Python3 implementation of Partial Least Squares Correlation, or [PLS](https://www.rotman-baycrest.on.ca/index.php?section=84).

## Status
[![Build Status](https://travis-ci.org/rmarkello/pyls.svg?branch=master)](https://travis-ci.org/rmarkello/pyls)
[![codecov](https://codecov.io/gh/rmarkello/pyls/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/pyls)

## Requirements
Python 3.5.X or above

See [`requirements.txt`](https://github.com/rmarkello/pyls/blob/master/requirements.txt) for more info on required modules.

## Installation
Using `git clone` and `python setup.py install` should do the trick.

## Usage
Currently both `BehavioralPLS` and `MeanCenteredPLS` are implemented; however, these are limited in their abilities (i.e., they can only deal with one grouping factor). Example usage:

```python
>>> import pyls

>>> rs = np.random.RandomState(123)
>>> X = rs.rand(20, 10000)
>>> Y = rs.rand(20, 10)
>>> opts = dict(n_perm=100, n_boot=50, n_split=50)
>>> bpls = pyls.BehavioralPLS(X, Y, seed=rs, **opts)

>>> groups = [1]*10 + [2]*10
>>> bpls2 = pyls.BehavioralPLS(X, Y, groups, seed=rs, **opts)
```
