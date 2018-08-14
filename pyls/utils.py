# -*- coding: utf-8 -*-

import numpy as np
import tqdm
from sklearn.utils import Bunch
from sklearn.utils.validation import check_random_state


class ResDict(Bunch):
    """
    Subclass of `sklearn.utils.Bunch` that only accepts keys in `cls.allowed`

    Also edits string representation to show non-empty keys
    """

    allowed = []

    def __init__(self, **kwargs):
        # only keep allowed keys
        i = {key: val for key, val in kwargs.items() if key in
             self.__class__.allowed}
        super().__init__(**i)

    def __str__(self):
        # override dict built-in string repr to display only non-empty keys
        items = [k for k in self.__class__.allowed if
                 (self.get(k) is not None and not _empty_dict(self.get(k)))]
        return '{name}({keys})'.format(name=self.__class__.__name__,
                                       keys=', '.join(items))

    def __setitem__(self, key, val):
        # legit we only want keys that are allowed
        if key in self.__class__.allowed:
            super().__setitem__(key, val)

    __repr__ = __str__


def _empty_dict(dobj):
    """
    Returns True if `dobj` is an empty dictionary; otherwise, returns False

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


def trange(n_iter, **kwargs):
    """
    Wrapper for :obj:`tqdm.trange` with some default options set

    Parameters
    ----------
    n_iter : int
        Number of iterations for progress bar

    Returns
    -------
    progbar : :obj:`tqdm.tqdm`
    """

    form = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
    defaults = dict(ascii=True, leave=False, bar_format=form)
    defaults.update(kwargs)

    return tqdm.trange(n_iter, **defaults)


def dummy_code(groups, n_cond=1):
    """
    Dummy codes `groups` and `n_cond`

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
    Generates group labels for `groups` and `n_cond`

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

    num_labels = len(groups * n_cond)

    return np.repeat(np.arange(num_labels) + 1, np.repeat(groups, n_cond))


def permute_cols(x, seed=None):
    """
    Permutes the rows for each column in `x` separately

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

    rs = check_random_state(seed)
    ix_i = rs.random_sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]
