=====================================
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

.. readme_installation:

Installation requirements
-------------------------
Currently, ``pyls`` works with Python 3.5+ and requires a few dependencies:

    - h5py
    - numpy
    - scikit-learn
    - scipy, and
    - tqdm

For detailed information on how to install ``pyls``, including these
dependencies, refer to our `installation instructions`_.

.. readme_development:

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

.. readme_licensing:

License Information
-------------------

This codebase is licensed under the GNU General Public License, version 2. The
full license can be found in the `LICENSE`_ file in the ``pyls`` distribution.

All trademarks referenced herein are property of their respective holders.

.. toctree::
   :maxdepth: 2

   installation
   usage
   api

.. |sparkles| replace:: ✨
.. _code of conduct: https://github.com/rmarkello/pyls/blob/master/CODE_OF_CONDUCT.md
.. _contributing guidelines: https://github.com/rmarkello/pyls/blob/master/CONTRIBUTING.md
.. _GitHub issues: https://github.com/rmarkello/pyls/issues
.. _installation instructions: https://pyls.readthedocs.io/en/latest/installation.html
.. _LICENSE: https://github.com/rmarkello/pyls/blob/master/LICENSE
