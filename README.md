pyls
----
A py3 implementation of [PLS](https://www.rotman-baycrest.on.ca/index.php?section=84). Not the whole thing, but just a few bits here and there.

## Status
[![Build Status](https://travis-ci.org/rmarkello/pyls.svg?branch=master)](https://travis-ci.org/rmarkello/pyls)
[![codecov](https://codecov.io/gh/rmarkello/pyls/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/pyls)

## Requirements
Only runs on Python 3.5 and above. See [`requirements.txt`](https://github.com/rmarkello/pyls/blob/master/requirements.txt) for more info on required modules.

## Installation
Using `git clone` and `python setup.py install` should do the trick.

## Usage
```
>>> from pyls import behavioral_pls
>>> brain = np.random.rand(20,10000)
>>> behavior = np.random.rand(20,10)
>>> bpls = behavioral_pls(brain, behavior)
>>> np.diag(bpls.d)  # singular values
array([ 36.84008605,  33.96990178, ..., 9.58797621])
>>> bpls.d_varexp    # % variance explained
array([ 0.23427363,  0.19919146, ...,  0.01586851])
>>> dir(bpls)        # all attributes/methods
['U', 'U_bci', 'U_bsr', ..., 'behav', 'brain']
```
