pyls: Partial Least Squares in Python
=====================================

This package provides a Python interface for performing partial least squares
(PLS) analyses.

.. image:: https://travis-ci.org/rmarkello/pyls.svg?branch=master
   :target: https://travis-ci.org/rmarkello/pyls
.. image:: https://circleci.com/gh/rmarkello/pyls.svg?style=shield
   :target: https://circleci.com/gh/rmarkello/pyls
.. image:: https://codecov.io/gh/rmarkello/pyls/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/rmarkello/pyls
.. image:: https://readthedocs.org/projects/pyls/badge/?version=latest
   :target: http://pyls.readthedocs.io/en/latest
.. image:: http://img.shields.io/badge/License-GPL%202.0-blue.svg
   :target: https://opensource.org/licenses/GPL-2.0

.. _readme_installation:

Installation requirements
-------------------------

Currently, ``pyls`` works with Python 3.5+ and requires a few dependencies:

    - h5py
    - numpy
    - scikit-learn
    - scipy, and
    - tqdm

Assuming you have the correct version of Python installed, you can install
``pyls`` by opening a terminal and running the following:

.. code-block:: bash

   git clone https://github.com/rmarkello/pyls.git
   cd pyls
   python setup.py install

All relevant dependencies will be installed alongside the ``pyls`` module.

.. _readme_quickstart:

Quickstart
----------

There are a number of ways to use ``pyls``, depending on the type of analysis
you would like to perform. Assuming you have two matrices ``X`` and ``Y``
representing different observations from a set of samples (i.e., subjects,
neurons, brain regions), you can run a simple analysis with:

.. code-block:: python

    >>> import pyls
    >>> results = pyls.behavioral_pls(X, Y)

For detailed information on the different methods available and how to
interpret the results object, please refer to our :ref:`user guide <usage>`.

.. _readme_development:

Development and getting involved
--------------------------------

If you've found a bug, are experiencing a problem, or have a question about
using the package, please head on over to our `GitHub issues`_ and make a new
issue with some information about it! Someone will try and get back to you
as quickly as possible, though please note that the primary developer for
``pyls`` (@rmarkello) is a graduate student so responses make take some time!

If you're interested in getting involved in the project: welcome |sparkles|!
We're thrilled to welcome new contributors. You should start by reading our
`code of conduct`_; all activity on ``pyls`` should adhere to the CoC. After
that, take a look at our `contributing guidelines`_ so you're familiar with the
processes we (generally) try to follow when making changes to the repository!
Once you're ready to jump in head on over to our issues to see if there's
anything you might like to work on.

.. _readme_licensing:

License Information
-------------------

This codebase is licensed under the GNU General Public License, version 2. The
full license can be found in the `LICENSE`_ file in the ``pyls`` distribution.

All trademarks referenced herein are property of their respective holders.

.. toctree::
   :maxdepth: 2

   usage
   api

.. |sparkles| replace:: ✨
.. _code of conduct: https://github.com/rmarkello/pyls/blob/master/CODE_OF_CONDUCT.md
.. _contributing guidelines: https://github.com/rmarkello/pyls/blob/master/CONTRIBUTING.md
.. _GitHub issues: https://github.com/rmarkello/pyls/issues
.. _LICENSE: https://github.com/rmarkello/pyls/blob/master/LICENSE
