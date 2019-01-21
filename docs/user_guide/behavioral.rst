.. _usage_behavioral:

Behavioral PLS
==============

What we call behavioral PLS in the ``pyls`` package is actually the more
traditional form of PLS, which attempts to find shared information between two
sets of variables. However, as with all things, there are a number of
ever-so-slightly different kinds of behavioral PLS that exist in the wild, so
to be thorough we're going to briefly explain the exact flavor implemented.

:py:func:`pyls.behavioral_pls` employs a symmetrical, singular value
decomposition (SVD) based form of PLS. It is sometimes referred to as
PLS-correlation or PLS-SVD. Notably, it is **not** the same as PLS regression.
That is, we are not assessing *dependent* relationships between sets of data,
but rather how the two sets generally covary.

To understand this a bit more we can walk through a quick example.

An exercise in calisthenics
---------------------------

Let's assume we have two matrices :math:`X` and :math:`Y`. For the sake of
working with something concrete we're going to use one of our example
datasets [1]_:

.. doctest::

    >>> import pyls.examples
    >>> data = pyls.examples.load_dataset('linnerud')
    >>> data
    PLSInputs(X, Y, n_perm, n_boot)

.. note::

    This is the same dataset as :py:func:`sklearn.datasets.load_linnerud`; the
    formatting has been lightly modified to better suit our purposes.

Looking at our matrices, we see:

.. doctest::

    >>> data.X.shape
    (20, 3)
    >>> data.X.head()
       Chins  Situps  Jumps
    0    5.0   162.0   60.0
    1    2.0   110.0   60.0
    2   12.0   101.0  101.0
    3   12.0   105.0   37.0
    4   13.0   155.0   58.0

The rows of our :math:`X` matrix here represent subjects, and the columns
indicate different types of exercises these subjects were able to perform.

.. doctest::

    >>> data.Y.shape
    (20, 3)
    >>> data.Y.head()
       Weight  Waist  Pulse
    0   191.0   36.0   50.0
    1   189.0   37.0   52.0
    2   193.0   38.0   58.0
    3   162.0   35.0   62.0
    4   189.0   35.0   46.0

The rows of our :math:`Y` matrix *also* represent subjects (critically, the
same subjects as in :math:`X`), and the columns indicate physiological
measurements taken for each subject. We can use behavioral PLS to establish
whether a relationship exists between the measured exercise and physiological
variables.

The cross-covariance matrix
---------------------------

Behavioral PLS works by decomposing the cross-covariance matrix, :math:`R`,
generated from the input matrices, where :math:`R = Y^{T} \times X`. The
results of PLS are a bit easier to interpret when :math:`R` is the
cross-correlation matrix instead of the cross-covariance matrix, which means
that we should z-score each feature in :math:`X` and :math:`Y` before
multiplying them; this is done automatically by :py:func:`pyls.behavioral_pls`
(but can be turned off by passing the ``covariance=True`` parameter).

In our example, :math:`R` ends up being a 3 x 3 matrix. Note that we pass
``norm=False`` to the cross-correlation function

.. doctest::

    >>> from pyls.compute import xcorr
    >>> R = xcorr(data.X, data.Y, norm=False)
    >>> R
    array([[-0.38969365, -0.49308365, -0.22629556],
           [-0.55223213, -0.64559803, -0.19149937],
           [ 0.15064802,  0.22503808,  0.03493306]])

Examining the first row, we can see that -0.3897 represents the correlation
between ``Chins`` and ``Weight`` across all the subjects, -0.4931 the
correlation between ``Situps`` and ``Weight``, and so on.

.. [1] Tenenhaus, M. (1998). La régression PLS: théorie et pratique. Editions
   technip.
