.. _usage_results:

PLS Results
===========

So you ran a PLS analysis and got some results. Congratulations! The easy part
is done. 🙃 Interpreting (trying to interpret) the results of a PLS
analysis---similar to interpreting the results of a PCA or factor analysis or
CCA or any other complex decomposition---can be difficult. The ``pyls`` package
contains some functions, tools, and data structures to try and help.

The :py:class:`~.structures.PLSResults` data structure is, at its core, a
Python dictionary that is designed to contain all possible results from any of
the analyses available in :py:mod:`pyls.types`. Let's generate a small example
results object to play around with. We'll use the dataset from the
:ref:`usage_behavioral` example:

.. doctest::

    >>> from pyls.examples import load_dataset
    >>> data = load_dataset('linnerud')

We can generate the results file by running the behavioral PLS analysis again.
We pass the ``verbose=False`` flag to suppress the progress bar that would
normally be displayed:

.. doctest::

    >>> from pyls import behavioral_pls
    >>> results = behavioral_pls(**data, verbose=False)
    >>> results
    PLSResults(x_weights, y_weights, x_scores, y_scores, y_loadings, singvals, varexp, permres, bootres, cvres, inputs)

Printing the ``results`` object gives us a helpful view of some of the
different outputs available to us. While we won't go into detail about all of
these (see the :ref:`ref_api` for info on those), we'll touch on a few of the
potentially more confusing ones.
