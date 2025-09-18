"""Utility functions for PLS toolbox."""

from contextlib import contextmanager

import numpy as np
import tqdm
from sklearn.utils import Bunch
from sklearn.utils.validation import check_array, check_random_state

try:
    from joblib import Parallel, delayed

    joblib_avail = True
except ImportError:
    joblib_avail = False


class ResDict(Bunch):
    """
    Subclass of `sklearn.utils.Bunch` that only accepts keys in `cls.allowed`.

    Also edits string representation to show non-empty keys
    """

    allowed = []

    def __init__(self, **kwargs):
        # only keep allowed keys
        i = {key: val for key, val in kwargs.items() if key in self.__class__.allowed}
        super().__init__(**i)

    def __str__(self):
        """Return string representation showing only non-empty keys."""
        items = [k for k in self.__class__.allowed if k in _not_empty_keys(self)]
        return "{name}({keys})".format(
            name=self.__class__.__name__, keys=", ".join(items)
        )

    def __setitem__(self, key, val):
        """Set item only if key is in the allowed list."""
        if key in self.__class__.allowed:
            super().__setitem__(key, val)

    def __eq__(self, value):
        """Return True if objects are equal, False otherwise."""
        # easy check -- are objects the same class?
        if not isinstance(value, self.__class__):
            return False
        # another easy check -- are the non-empty keys different?
        if _not_empty_keys(self) != _not_empty_keys(value):
            return False
        # harder check -- iterate through everything and check item equality
        # potentially recursive checks if sub-items are dictionaries
        for k, v in self.items():
            v2 = value.get(k, None)
            if v is None and v2 is None:
                continue
            # recursive dictionary comparison
            if isinstance(v, dict) and isinstance(v2, dict):
                if v != v2:
                    return False
            # compare using numpy testing suite
            # this is because arrays may be different size and numpy testing
            # is way more solid than anything we could come up with
            else:
                try:
                    np.testing.assert_array_almost_equal(v, v2)
                except (TypeError, AssertionError):
                    return False

        return True

    def __ne__(self, value):
        """Return True if objects are not equal, False otherwise."""
        return not self == value

    __repr__ = __str__


def _not_empty_keys(dictionary):
    """
    Return list of non-empty keys in `dictionary`.

    Non-empty keys are defined as (1) not being None-type and (2) not being an
    empty dictionary, itself

    Parameters
    ----------
    dictionary : dict
        Object to query for non-empty keys

    Returns
    -------
    keys : list
        Non-empty keys in `dictionary`
    """
    if not isinstance(dictionary, dict):
        raise TypeError(
            "Provided input must be type dict, not {}".format(type(dictionary))
        )

    keys = []
    for key, value in dictionary.items():
        if value is not None and not _empty_dict(value):
            keys.append(key)

    return set(keys)


def _empty_dict(dobj):
    """
    Return True if `dobj` is an empty dictionary; otherwise, returns False.

    Parameters
    ----------
    dobj
        Any Python object

    Returns
    -------
    empty : bool
        Whether `dobj` is an empty dictionary-like object
    """
    try:
        return len(dobj.keys()) == 0
    except (AttributeError, TypeError):
        return False


def trange(n_iter, verbose=True, **kwargs):
    """
    Wrap :obj:`tqdm.trange` with some default options set.

    Parameters
    ----------
    n_iter : int
        Number of iterations for progress bar
    verbose : bool, optional
        Whether to return an :obj:`tqdm.tqdm` progress bar instead of a range
        generator. Default: True
    kwargs
        Key-value arguments provided to :func:`tqdm.trange`

    Returns
    -------
    progbar : :obj:`tqdm.tqdm`
    """
    form = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} | {elapsed}<{remaining}"
    )
    defaults = dict(ascii=True, leave=False, bar_format=form)
    defaults.update(kwargs)

    return tqdm.trange(n_iter, disable=not verbose, **defaults)


def dummy_code(groups, n_cond=1):
    """
    Dummycode `groups` and `n_cond`.

    Parameters
    ----------
    groups : (G,) list
        List with number of subjects in each of `G` groups
    n_cond : int, optional
        Number of conditions, for each subject. Default: 1

    Returns
    -------
    Y : (S, F) `numpy.ndarray`
        Dummy-coded group array
    """
    labels = dummy_label(groups, n_cond)
    dummy = np.column_stack([labels == g for g in np.unique(labels)])

    return dummy.astype(int)


def dummy_label(groups, n_cond=1):
    """
    Generate group labels for `groups` and `n_cond`.

    Parameters
    ----------
    groups : (G,) list
        List with number of subjects in each of `G` groups
    n_cond : int, optional
        Number of conditions, for each subject. Default: 1

    Returns
    -------
    Y : (S,) `numpy.ndarray`
        Dummy-label group array
    """
    num_labels = len(groups) * n_cond

    return np.repeat(np.arange(num_labels) + 1, np.repeat(groups, n_cond))


def permute_cols(x, seed=None):
    """
    Permute the rows for each column in `x` separately.

    Taken from https://stackoverflow.com/a/27489131

    Parameters
    ----------
    x : (S, B) array_like
        Input array to be permuted
    seed : {int, :obj:`numpy.random.RandomState`, None}, optional
        Seed for random number generation. Default: None

    Returns
    -------
    permuted : `numpy.ndarray`
        Permuted array
    """
    # can't permute row with only 1 sample...
    x = check_array(x)
    rs = check_random_state(seed)
    ix_i = rs.random_sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]


class _unravel:
    """
    Small utility to unravel generator object into a list.

    Parameters
    ----------
    x : generator

    Returns
    -------
    y : list
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return [f for f in x]

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass


@contextmanager
def get_par_func(n_proc, func, **kwargs):
    """
    Create joblib-style parallelization function if joblib is available.

    Parameters
    ----------
    n_proc : int
        Number of processors (i.e., jobs) to use for parallelization
    func : function
        Function to parallelize

    Returns
    -------
    parallel : :obj:`joblib.Parallel` object
        Object to parallelize over `func`
    func : :obj:`joblib.delayed` object
        Provided `func` wrapped in `joblib.delayed`
    """
    if joblib_avail:
        func = delayed(func)
        with Parallel(
            n_jobs=n_proc, max_nbytes=1e6, mmap_mode="r+", **kwargs
        ) as parallel:
            yield parallel, func
    else:
        parallel = _unravel()
        yield parallel, func
