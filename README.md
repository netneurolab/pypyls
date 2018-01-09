# pyls
A py3 implementation of Partial Least Squares Correlation, or [PLS](https://www.rotman-baycrest.on.ca/index.php?section=84).

## Status
[![Build Status](https://travis-ci.org/rmarkello/pyls.svg?branch=master)](https://travis-ci.org/rmarkello/pyls)
[![codecov](https://codecov.io/gh/rmarkello/pyls/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/pyls)

## Requirements
This only runs on Python 3.5 and above, solely because it makes use of the `@` operator for matrix multiplication.

See [`requirements.txt`](https://github.com/rmarkello/pyls/blob/master/requirements.txt) for more info on required modules.

## Installation
Using `git clone` and `python setup.py install` should do the trick.

## Usage
```
>>> import pyls

>>> rs = np.random.RandomState(123)
>>> X = rs.rand(20, 10000)
>>> Y = rs.rand(20, 10)
>>> bpls = pyls.BehavioralPLS(X, Y, n_perm=100, n_boot=50, n_split=50)

# list the singular values from the decomposition
>>> np.diag(bpls.d)
array([ 36.84008605,  33.96990178, ..., 9.58797621])

# list the percent variance explained by each component
>>> bpls.d_varexp
array([ 0.23427363,  0.19919146, ...,  0.01586851])

# all the attributes available
>>> dir(bpls)
['U', 'U_bci', 'U_bsr', ..., 'behav', 'brain']
```

This version of PLS has split-half resampling implenented; simply set `n_split` when instantiating `BehavioralPLS`.
