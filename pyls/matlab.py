# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio


def import_matlab_result(fname):
    """
    Imports ``fname`` PLS result from matlab

    Parameters
    ----------
    fname : str
        Filepath to result.mat file output by Matlab PLS

    Returns
    -------
    result : dict
        Matlab PLS result dictionary
    """

    result = sio.loadmat(fname)['result'][0, 0]

    labels = list(result.dtype.fields.keys())
    result = {labels[n]: value for n, value in enumerate(result)}

    to_dict = ['perm_result', 'boot_result', 'other_input',  'perm_splithalf']
    for attr in to_dict:
        if result.get(attr, None) is not None:
            labels = list(result[attr].dtype.fields.keys())
            result[attr] = {labels[n]: np.squeeze(value) for n, value in
                            enumerate(result[attr][0, 0])}

    for key, val in result.items():
        if isinstance(val, np.ndarray):
            result[key] = np.squeeze(val)

    return result
