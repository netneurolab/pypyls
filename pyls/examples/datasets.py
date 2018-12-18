# -*- coding: utf-8 -*-
"""
Functions and utilities for getting datasets for PLS examples
"""

import json
import os
from pkg_resources import resource_filename
import requests
import urllib

import numpy as np
import pandas as pd

from ..structures import PLSInputs

with open(resource_filename('pyls', 'examples/datasets.json'), 'r') as src:
    _DATASETS = json.load(src)


def available_datasets():
    """
    Lists available datasets to download

    Returns
    -------
    datasets : list
        List of available datasets
    """

    return list(_DATASETS.keys())


def _get_data_dir(data_dir=None):
    """
    Gets path to pyls data directory

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'PYLS_DATA'; if that is not set, will use
        `~/pyls-data` instead. Default: None

    Returns
    -------
    data_dir : str
        Path to use as data directory
    """

    if data_dir is None:
        data_dir = os.environ.get('PYLS_DATA', os.path.join('~', 'pyls-data'))
    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir


def load_dataset(name, data_dir=None, verbose=1):
    """
    Loads dataset provided by `name` into a :obj:`PLSInputs` object

    Parameters
    ----------
    name : str
        Name of dataset. Must be one of the datasets listed in
        :func:`pyls.examples.available_datasets()`
    data_dir : str, optional
        Path to use as data directory to store dataset. If not specified, will
        check for environmental variable 'PYLS_DATA'; if that is not set, will
        use `~/pyls-data` instead. Default: None
    verbose : int, optional
        Level of verbosity for status messages about fetching/loading dataset.
        Set to 0 for no updates. Default: 1

    Returns
    -------
    dataset : :obj:`~.structures.PLSInputs`
        PLSInputs object containing pre-loaded data ready to run PLS analysis.
        Rerun the analysis by calling :func:`pyls.behavioral_pls(**dataset)` or
        :func:`pyls.meancentered_pls(**dataset)`, as appropriate
    """

    if name not in available_datasets():
        raise ValueError('Provided dataset {} not available. Must be one of {}'
                         .format(name, available_datasets()))

    data_path = os.path.join(_get_data_dir(data_dir), name)
    _get_dataset(name, data_path, verbose=verbose)

    dataset = PLSInputs()
    for key, value in _DATASETS.get(name, {}).items():
        if isinstance(value, str):
            fname = os.path.join(data_path, value)
            if fname.endswith('.csv'):
                value = pd.read_csv(fname, index_col=0)
            elif fname.endswith('.txt'):
                value = np.loadtxt(fname)
            elif fname.endswith('.npy'):
                value = np.load(fname)
            else:
                raise ValueError('Cannot recognize datatype of {}. Please '
                                 'create an issue on GitHub with dataset you '
                                 'are trying to load ({})'.format(fname, name))
        dataset[key] = value

    # make some dataset-specific corrections
    if name == 'whitaker_vertes_2016':
        dataset.X = dataset.X.T

    return dataset


def _get_dataset(name, data_dir, verbose=1):
    """
    Downloads dataset defined by `name`

    Parameters
    ----------
    name : str
        Name of dataset. Must be one of the datasets listed in
        :func:`pyls.examples.available_datasets()`
    data_dir : str
        Path to use as data directory to store dataset
    """

    os.makedirs(data_dir, exist_ok=True)

    for url in _DATASETS.get(name, {}).get('urls', []):
        parse = urllib.parse.urlparse(url)
        fname = os.path.join(data_dir, os.path.basename(parse.path))

        if not os.path.exists(fname):
            out = requests.get(url)
            with open(fname, 'wb') as dest:
                dest.write(out.content)
