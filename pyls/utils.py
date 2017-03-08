#!/usr/bin/env python

import glob
import numpy as np

def get_rand_list(globform, num_runs=2, seed=None):
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
    num_runs : int
        number of runs for given subject (will be split evenly b/w datasets)
    seed : int
        for seeding random choice function to ensure replicability

    Returns
    -------
    list, list : test, retest datasets
    """

    if seed is not None: np.random.seed(seed)

    l = sorted(glob.glob(globform, recursive=True))

    booll  = np.zeros(len(l)//num_runs,dtype='bool')
    booll[np.random.choice(np.arange(len(l)//num_runs),
                           size=len(l)//(num_runs/2),
                           replace=False)] = True

    first  = np.array(l[slice(0,len(l),num_runs)])
    second = np.array(l[slice(1,len(l),num_runs)])

    test   = sorted(list(first[booll])  + list(second[~booll]))
    retest = sorted(list(first[~booll]) + list(second[booll]))

    return test, retest
