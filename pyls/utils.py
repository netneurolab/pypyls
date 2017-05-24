#!/usr/bin/env python

from collections import Counter
import numpy as np
import nibabel as nib


def flatten_niis(fnames,thresh=0.2):
    """
    Loads nii files in `fnames`, flattens, and removes non-zero voxels

    Parameters
    ----------
    fnames : list (N)
        list of nii files to be loaded
    thresh : float
        determines # of participants that must have activity in voxel for it
        to be kept in final array

    Returns
    -------
    array: N x non-zero voxels
    """

    # get some information on the data
    cutoff = np.ceil(thresh*len(fnames))
    x, y, z = nib.load(fnames[0]).shape
    all_data = np.zeros((len(fnames),x*y*z))

    # load in data
    for n, f in enumerate(fnames):
        temp = nib.load(f).get_data().flatten()
        all_data[n] = temp

    # get non-zero voxels
    nz = np.array(Counter(np.where(all_data!=0)[1]).most_common())
    all_data = all_data[:,nz[np.where(nz[:,1]>cutoff)[0],0]]

    return all_data


def xcorr(X, Y):
    """
    Calculates the cross correlation of `X` and `Y`

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

    avg,  stdev  = X.mean(axis=0), X.std(axis=0)
    davg, dstdev = Y.mean(axis=0),  Y.std(axis=0)

    checknan = np.where(stdev==0)[0]
    if checknan.size > 0:
        X[:,checknan], avg[checknan], stdev[checknan] = 0, 0, 1

    dchecknan = np.where(dstdev==0)[0]
    if dchecknan.size > 0:
        Y[:,dchecknan], davg[dchecknan], dstdev[dchecknan] = 0, 0, 1

    X, Y = (X-avg)/stdev, (Y-davg)/dstdev
    xprod = (Y.T @ X)/(X.shape[0]-1)

    return xprod


def normalize(mat, dim=0):
    """
    Normalizes `origin` along dimension `dim`

    Parameters
    ----------
    mat : array
    dim : bool
        Dimension for normalization

    Returns
    -------
    array : normalized `mat`
    """

    normal_base = np.linalg.norm(mat, axis=dim, keepdims=True)
    if dim == 1: normal_base = normal_base.T  # to ensure proper broadcasting

    # to avoid DivideByZero errors
    zero_items = np.where(normal_base==0)
    normal_base[zero_items] = 1

    # normalize and re-set zero_items
    normal = mat/normal_base
    normal[zero_items] = 0

    return normal
