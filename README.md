# pyls

This package provides a Python interface for partial least squares (PLS) analysis, a multivariate statistical technique used to relate two sets of variables.

[![Build Status](https://travis-ci.org/rmarkello/pyls.svg?branch=master)](https://travis-ci.org/rmarkello/pyls)
[![CircleCI](https://circleci.com/gh/rmarkello/pyls.svg?style=shield)](https://circleci.com/gh/rmarkello/pyls)
[![Codecov](https://codecov.io/gh/rmarkello/pyls/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/pyls)
[![Documentation Status](https://readthedocs.org/projects/pyls/badge/?version=latest)](http://pyls.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-GPL%202.0-blue.svg)](https://opensource.org/licenses/GPL-2.0)

## Table of Contents

If you know where you're going, feel free to jump ahead:

* [Installation and setup](#requirements-and-installation)
* [Purpose](#purpose)
  * [Overview](#overview)
  * [Background](#background)
* [Usage](#usage)
  * [PLS correlation methods](#pls-correlation-methods)
    * [Behavioral PLS](#behavioral-pls)
    * [Mean-centered PLS](#mean-centered-pls)
  * [PLS regression methods](#pls-regression-methods)
    * [Regression with SIMPLS](#regression-with-simpls)
  * [PLS results](#results)
  
## Installation and setup

This package requires Python >= 3.5. Assuming you have the correct version of Python installed, you can install this package by opening a terminal and running the following:

```bash
git clone https://github.com/rmarkello/pyls.git
cd pyls
python setup.py install
```

There are plans (hopes?) to get this set up on PyPI for an easier installation process, but that is a long-term goal!

## Purpose

### Overview

Partial least squares (PLS) is a statistical technique that aims to find shared information between two sets of variables. 
If you're unfamiliar with PLS and are interested in a thorough (albeit quite technical) treatment of it [Abdi et al., 2013](https://doi.org/10.1007/978-1-62703-059-5_23) is a good resource.
There are multiple "flavors" of PLS that are tailored to different use cases; this package implements two functions that fall within the category typically referred to as **PLS-C** (PLS correlation) or **PLS-SVD** (PLS singular value decomposition) and one function that falls within the category typically referred to as **PLS-R** (PLS regression).

### Background

The functionality of the current package largely mirrors that originally introduced by [McIntosh et al., (1996)](https://www.ncbi.nlm.nih.gov/pubmed/9345485) in their [Matlab toolbox](https://www.rotman-baycrest.on.ca/index.php?section=84).
However, while the Matlab toolbox has a significant number of tools dedicated to integrating neuroimaging-specific paradigms (i.e., loading M/EEG and fMRI data), the current Python package aims to implement and expand on only the core _statistical_ functions of that toolbox.

While the core algorithms of PLS implemented in this package are present (to a degree) in [`scikit-learn`](`https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cross_decomposition`), this package provides a different API and includes some additional functionality.
Namely, `pyls`:

1. Has integrated significance and reliability testing via built-in permutation testing and bootstrap resampling,
2. Implements [mean-centered PLS](https://www.ncbi.nlm.nih.gov/pubmed/20656037) for multivariate group/condition comparisons,
3. Uses the [SIMPLS](https://doi.org/10.1016%2F0169-7439%2893%2985002-X) instead of the [NIPALS algorithm](https://doi.org/10.1016/B978-0-12-426653-7.50032-6) for PLS regression

## Usage

`pyls` implement two subtypes of PLS-C: a more traditional form that we call "behavioral PLS" (`pyls.behavioral_pls`) and a somewhat newer form that we call "mean-centered PLS" (`pyls.meancentered_pls`).
It also implements one type of PLS-R, which uses the SIMPLS algorithm (`pyls.pls_regression`); this is, in principle, very similar to "behavioral PLS."

### PLS correlation methods

#### Behavioral PLS

As the more "traditional" form of PLS-C, `pyls.behavioral_pls` looks to find relationships between two sets of variables. 
To run a behavioral PLS we would do the following:

```python
>>> import numpy as np

# let's create two data arrays with 80 observations
>>> X = np.random.rand(80, 10000)  # a 10000-feature (e.g., neural) data array
>>> Y = np.random.rand(80, 10)     # a 10-feature (e.g., behavioral) data array

# we're going to pretend that this data is from 2 groups of 20 subjects each,
# and that each subject participated in 2 task conditions
>>> groups = [20, 20]  # a list with the number of subjects in each group
>>> n_cond = 2         # the number of tasks or conditions

# run the analysis and look at the results structure
>>> from pyls import behavioral_pls
>>> bpls = behavioral_pls(X, Y, groups=groups, n_cond=n_cond)
>>> bpls
PLSResults(x_weights, y_weights, x_scores, y_scores, y_loadings, singvals, varexp, permres, 
bootres, splitres, cvres, inputs)
```

#### Mean-centered PLS

In contrast to behavioral PLS, `pyls.meancentered_pls` doesn't look to find relationships between two sets of variables, but rather tries to find relationships between _groupings_ in a single set of variables. As such, we will only provide it with _one_ of our created data arrays (`X`) and it will attempt to examine how the features of that array differ between groups and/or conditions. To run a mean-centered PLS we would do the following:

```python
>>> from pyls import meancentered_pls
>>> mpls = meancentered_pls(X, groups=groups, n_cond=n_cond)
>>> mpls
PLSResults(x_weights, y_weights, x_scores, y_scores, singvals, varexp, permres, bootres, splitres,
inputs)
```

### PLS regression methods

#### Regression with SIMPLS

Whereas `pyls.behavioral_pls` aims to maximize the symmetric relationship between `X` and `Y`, `pyls.pls_regression` performs a directed decomposition.
That is, it aims to find components in `X` that explain the most variance in `Y` (but not necessarily vice versa).
To run a PLS regression analysis we would do the following:

```python
>>> from pyls import pls_regression
>>> plsr = pls_regression(X, Y, n_components=5)
>>> plsr
PLSResults(x_weights, x_scores, y_scores, y_loadings, varexp, permres, bootres, inputs)
```

Currently `pyls.pls_regression()` does not support groups or conditions.

### PLS Results

The docstrings of the results objects (`bpls`, `plsr`, and `mpls` in the above example) have some information describing what each output represents, so while we work on improving our documentation you can rely on those for some insight! Try typing `help(bpls)`, `help(plsr)`, or `help(mpls)` to get more information on what the different values represent.

If you are at all familiar with the Matlab PLS toolbox you might notice that the results structures have a dramatically different naming convention; despite this all the same information should be present!
