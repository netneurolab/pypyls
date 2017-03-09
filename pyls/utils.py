#!/usr/bin/env python

import glob
import numpy as np

def get_rand_list(globform, seed=None):
    """
    Function for use in randomizing test/retest datasets

    Applicable only if each subject has two runs. Given a globform,
    will create a test/retest dataset that is balanced to ensure even inclusion
    of different runs (i.e., test dataset will be half first runs/half second
    runs and vice versa for retest).

    Parameters
    ----------
    globform : str
        used to find relevant files with glob.glob
    seed : int
        for seeding random choice function to ensure replicability

    Returns
    -------
    list, list : test, retest datasets
    """

    if seed is not None: np.random.seed(seed)

    l = sorted(glob.glob(globform, recursive=True))

    booll  = np.zeros(len(l)//2,dtype='bool')
    booll[np.random.choice(np.arange(len(l)//2),
                           size=len(l)//(2/2),
                           replace=False)] = True

    first  = np.array(l[slice(0,len(l),2)])
    second = np.array(l[slice(1,len(l),2)])

    test   = sorted(list(first[booll])  + list(second[~booll]))
    retest = sorted(list(first[~booll]) + list(second[booll]))

    return test, retest
