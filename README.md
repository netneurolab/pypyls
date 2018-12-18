# pyls

This package provides a Python interface for partial least squares (PLS) analysis, a multivariate statistical technique used to relate two sets of variables.

[![Build Status](https://travis-ci.org/rmarkello/pyls.svg?branch=master)](https://travis-ci.org/rmarkello/pyls)
[![CircleCI](https://circleci.com/gh/rmarkello/pyls.svg?style=shield)](https://circleci.com/gh/rmarkello/pyls)
[![Codecov](https://codecov.io/gh/rmarkello/pyls/branch/master/graph/badge.svg)](https://codecov.io/gh/rmarkello/pyls)
[![Documentation Status](https://readthedocs.org/projects/pyls/badge/?version=latest)](http://pyls.readthedocs.io/en/latest/?badge=latest)

## Table of Contents

If you know where you're going, feel free to jump ahead:

* [Installation](#requirements-and-installation)
* [Purpose](#purpose)
  * [Overview](#overview)
  * [Background](#background)
  * [Development](#development)
* [Example usage](#usage)
  * [Behavioral PLS](#behavioral-pls)
  * [Mean-centered PLS](#mean-centered-pls)
  * [PLS results](#results)
* [How to get involved](#how-to-get-involved)

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

Partial least squares (PLS) is a statistical technique that aims to find shared information between two sets of variables. If you're unfamiliar with PLS and are interested in a thorough (albeit quite technical) treatment of it [Abdi et al., 2013](https://doi.org/10.1007/978-1-62703-059-5_23) is a good resource. There are multiple "flavors" of PLS that are tailored to different use cases, but the implementation in the current package is often referred to as **PLS-C** (PLS correlation) or **PLS-SVD** (PLS singular value decomposition).

### Background

The functionality of the current package largely mirrors that originally introduced by [McIntosh et al., (1996)](https://www.ncbi.nlm.nih.gov/pubmed/9345485) in their [Matlab toolbox](https://www.rotman-baycrest.on.ca/index.php?section=84). However, while the Matlab toolbox has a significant number of tools dedicated to integrating neuroimaging-specific paradigms (i.e., loading M/EEG and fMRI data), the current Python package aims to implement and expand on only the core _statistical_ functions of that toolbox.

While the core algorithm of PLS implemented in this package is also present in [`scikit-learn`](`http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html`), this package provides a slightly different API and includes some additional functionality. Namely, `pyls`:

1. Has integrated significance and reliability testing via built-in permutation testing and bootstrap resampling,
2. Implements [mean-centered PLS](https://www.ncbi.nlm.nih.gov/pubmed/20656037) for multivariate group/condition comparisons.

### Development

This package has largely been developed in the spare time of a single graduate student ([`@rmarkello`](https://github.com/rmarkello)), so while it would be :sparkles: amazing :sparkles: if anyone else finds it helpful, this package is not currently accepting requests for new features.

## Usage

`pyls` implement two subtypes of PLS-C: a more traditional form that we call "behavioral PLS" (accessible as the function `behavioral_pls`) and a somewhat newer form that we call "mean-centered PLS" (accessible as the function `meancentered_pls`).

### Behavioral PLS

As the more "traditional" form of PLS-C, `behavioral_pls` looks to find relationships between two sets of variables. To run a behavioral PLS we would do the following:

```python
>>> from pyls import behavioral_pls

# let's create two data arrays with 80 observations
>>> X = np.random.rand(80, 10000)  # a 10000-feature (e.g., neural) data array
>>> Y = np.random.rand(80, 10)     # a 10-feature (e.g., behavioral) data array

# we're going to pretend that this data is from 2 groups of 20 subjects each,
# and that each subject participated in 2 task conditions
>>> groups = [20, 20]  # a list with the number of subjects in each group
>>> n_cond = 2         # the number of tasks or conditions

# run the analysis and look at the results structure
>>> bpls = behavioral_pls(X, Y, groups=groups, n_cond=n_cond)
>>> bpls
PLSResults(u, s, v, brainscores, behavscores, behavcorr, permres, bootres, splitres, cvres, inputs)
```

### Mean-centered PLS

In contrast to behavioral PLS, `meancentered_pls` doesn't look to find relationships between two sets of variables, but rather tries to find relationships between _groupings_ in a single set of variables. As such, we will only provide it with _one_ of our created data arrays (`X`) and it will attempt to examine how the features of that array differ between groups and/or conditions. To run a mean-centered PLS we would do the following:

```python
>>> from pyls import meancentered_pls
>>> mpls = pyls.meancentered_pls(X, groups=groups, n_cond=n_cond)
>>> mpls
PLSResults(u, s, v, brainscores, brainscores_dm, designscores, permres, bootres, splitres, inputs)
```

### Results

The doc-strings of the results objects (`bpls` and `mpls` in the above example) have some information describing what each output represents, so while we work on [getting some better documentation](https://github.com/rmarkello/pyls/issues/19) you can rely on those for some insight! Try typing `help(bpls)` or `help(mpls)` to get more information on what the different values represent.

If you are at all familiar with the Matlab PLS toolbox you might notice that the results structures have a differeng naming convention; despite this all the same information should be present!

## How to get involved

We're thrilled to welcome new contributors! If you're interesting in getting involved, you should start by reading our [contributing guidelines](https://github.com/rmarkello/pyls/blob/master/CONTRIBUTING.md) and [code of conduct](https://github.com/rmarkello/pyls/blob/master/Code_of_Conduct.md).

Once you're done with that, you can take a look at our [roadmap](https://github.com/rmarkello/pyls/issues/26) and the accompanying [project](https://github.com/rmarkello/pyls/projects/1). If you find something you'd like to work on, head on over to the relevant [issue](https://github.com/rmarkello/pyls/issues) and let us know.

If you've found a bug, are experiencing a problem, or have a question, create a new issue with some information about it!
