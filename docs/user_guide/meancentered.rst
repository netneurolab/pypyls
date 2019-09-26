.. _usage_meancentered:

Mean-centered PLS
=================

In contrast to behavioral PLS, mean-centered PLS doesn't aim to find
relationships between two sets of variables. Instead, it tries to find
relationships between *groupings* in a single set of variables. Indeed, you can
think of it almost like a multivariate t-test or ANOVA (depending on how many
groups you have).

An oenological example
----------------------

.. doctest::

    >>> from pyls.examples import load_dataset
    >>> data = load_dataset('wine')

This is the same dataset as in :py:func:`sklearn.datasets.load_wine`; the
formatting has just been lightly modified to better suit our purposes.

Our ``data`` object can be treated as a dictionary, containing all the
information necessary to run a PLS analysis. The keys can be accessed as
attributes, so we can take a quick look at our input matrix:

.. doctest::

    >>> sorted(data.keys())
    ['X', 'groups', 'n_boot', 'n_perm']
    >>> data.X.shape
    (178, 13)
    >>> data.X.columns
    Index(['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
           'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
           'proanthocyanins', 'color_intensity', 'hue',
           'od280/od315_of_diluted_wines', 'proline'],
          dtype='object')
    >>> data.groups
    [59, 71, 48]
