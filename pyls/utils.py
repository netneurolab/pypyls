#!/usr/bin/env python

from collections import Counter

import nibabel as nib
import numpy as np


def flatten_niis(fnames, thresh=0.2):
    """
    Loads 3D nii files in `fnames`, flattens, and removes non-zero voxels

    Parameters
    ----------
    fnames : list
        List of nifti files to be loaded, flattened, and masked (via
        thresholding).
    thresh : float, optional
        Threshold to determine # of participants that must have activity in a
        voxel for it to be kept in final array. Must be within (0,1)
        (default: 0.2).

    Returns
    -------
    array
        Stacked, masked input data (N x voxels)
    """

    if thresh > 1 or thresh < 0:
        raise ValueError("Thresh must be between 0 and 1.")

    # get some information on the data
    cutoff   = np.ceil(thresh * len(fnames))
    shape    = nib.load(fnames[0]).shape
    all_data = np.zeros((len(fnames), np.product(shape[:3])))

    # load in data
    for n, f in enumerate(fnames):
        temp = nib.load(f).get_data()
        if temp.ndim > 3: temp = temp[:, :, :, 0]
        all_data[n] = temp.flatten()

    # get non-zero voxels
    non_zero = np.array(Counter(all_data.nonzero()[1]).most_common())
    all_data = all_data[:, non_zero[np.where(non_zero[:, 1] > cutoff)[0], 0]]

    return all_data


def xcorr(X, Y):
    """
    Calculates the cross-correlation of `X` and `Y`

    Parameters
    ----------
    X : array_like
        Data array
    Y : array_like
        Behavior array

    Returns
    -------
    array
        Cross-correlation of `X` and `Y`
    """

    Xz, Yz = np.nan_to_num(zscore(X)), np.nan_to_num(zscore(Y))
    xprod = (Yz.T @ Xz) / (Xz.shape[0] - 1)

    return xprod


def zscore(X):
    """
    Z-scores `X` by subtracting mean, dividing by standard deviation

    Performs columnwise (not rowwise) normalization. If the standard deviation
    of any column of X == 0, that column is returned unchanged

    Parameters
    ----------
    X : array_like
        Two-dimensional input array

    Returns
    -------
    array
        Z-scored input
    """

    arr = np.asarray(X.copy())

    avg, stdev = arr.mean(axis=0), arr.std(axis=0)
    zero_items = np.where(stdev == 0)[0]

    if zero_items.size > 0:
        avg[zero_items], stdev[zero_items] = 0, 1

    zarr = (arr - avg) / stdev
    zarr[:, zero_items] = arr[:, zero_items]

    return zarr


def normalize(X, dim=0):
    """
    Normalizes `X` along dimension `dim`

    Utilizes Frobenius norm (or Hilbert-Schmidt norm, L_{p,q} norm where p=q=2)

    Parameters
    ----------
    X : array_like
    dim : bool
        Dimension for normalization

    Returns
    -------
    array
        Normalized `X`
    """

    normed = X.copy()

    normal_base = np.linalg.norm(normed, axis=dim, keepdims=True)
    if dim == 1: normal_base = normal_base.T  # to ensure proper broadcasting

    # to avoid DivideByZero errors
    zero_items = np.where(normal_base == 0)
    normal_base[zero_items] = 1

    # normalize and re-set zero_items
    normed /= normal_base
    normed[zero_items] = 0

    return normed
