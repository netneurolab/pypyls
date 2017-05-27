#!/usr/bin/env python

from collections import Counter
import numpy as np
import nibabel as nib


def flatten_niis(fnames, thresh=0.2):
    """
    Loads 3D nii files in `fnames`, flattens, and removes non-zero voxels

    Parameters
    ----------
    fnames : list (N)
        list of nii files to be loaded
    thresh : float [0,1]
        determines # of participants that must have activity in voxel for it
        to be kept in final array

    Returns
    -------
    array: N x non-zero voxels
    """

    if thresh > 1 or thresh < 0:
        raise ValueError("Thresh must be between 0 and 1.")

    # get some information on the data
    cutoff = np.ceil(thresh*len(fnames))
    shape = nib.load(fnames[0]).shape
    all_data = np.zeros((len(fnames), np.product(shape[:3])))

    # load in data
    for n, f in enumerate(fnames):
        temp = nib.load(f).get_data()
        if temp.ndim > 3: temp = temp[:,:,:,0]
        all_data[n] = temp.flatten()

    # get non-zero voxels
    non_zero = np.array(Counter(np.where(all_data!=0)[1]).most_common())
    all_data = all_data[:,non_zero[np.where(non_zero[:,1]>cutoff)[0],0]]

    return all_data


def xcorr(X, Y):
    """
    Calculates the cross-correlation of `X` and `Y`

    Parameters
    ----------
    X : array
        data array
    Y : array
        behavior array

    Returns
    -------
    array : cross-correlation of `X` and `Y`
    """

    X, Y = zscore(X), zscore(Y)
    xprod = (Y.T @ X)/(X.shape[0]-1)

    return xprod


def zscore(X):
    """
    Z-scores `X` by subtracting mean, dividing by standard deviation

    Parameters
    ----------
    X : array

    Returns
    -------
    array : z-scored input
    """

    avg, stdev = X.mean(axis=0), X.std(axis=0)

    zero_items = np.where(stdev==0)[0]
    if zero_items.size > 0:
        X[:,zero_items], avg[zero_items], stdev[zero_items] = 0, 0, 1

    return (X-avg)/stdev


def normalize(X, dim=0):
    """
    Normalizes `X` along dimension `dim`

    Utilizes Frobenius norm (or Hilbert-Schmidt norm, L_p,q norm where p=q=2)

    Parameters
    ----------
    X : array
    dim : bool
        dimension for normalization

    Returns
    -------
    array : normalized `X`
    """

    normal_base = np.linalg.norm(X, axis=dim, keepdims=True)
    if dim == 1: normal_base = normal_base.T  # to ensure proper broadcasting

    # to avoid DivideByZero errors
    zero_items = np.where(normal_base==0)
    normal_base[zero_items] = 1

    # normalize and re-set zero_items
    normal = X/normal_base
    normal[zero_items] = 0

    return normal
