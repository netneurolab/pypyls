# -*- coding: utf-8 -*-
"""
Utilities for plotting functions
"""

import numpy as np
import pandas as pd


def _set_group_lvls(n_conds, n_grps, grp_lvls=None):
    """
    Derives a pandas data series of group labels

    Parameters
    ----------
    n_conds : int
        Number of conditions in the analysis
    n_grps : int
        Number of groups in the analysis
    grp_lvls : list, optional
        List of group labels

    Returns
    -------
    labels : pd.Series
        Series of group labels aligned to the input data structure
    """

    grping = []
    if grp_lvls is None:
        for i in range(n_grps):
            grping += ['Group ' + str(i)] * n_conds
    else:
        for i in range(n_grps):
            grping.extend([grp_lvls[i]] * n_conds)
    return pd.Series(grping, name='Group')


def _set_cond_lvls(n_conds, n_grps, cond_lvls=None):
    """
    Derives a pandas series of condition labels

    Parameters
    ----------
    n_conds : int
        Number of conditions in the analysis
    n_grps : int
        Number of groups in the analysis
    cond_lvls : list, optional
        List of condition labels

    Returns
    -------
    labels : pd.Series
        Series of condition labels aligned to the input data structure
    """

    if cond_lvls is None:
        cond_lvls = ['Condition ' + str(i) for i in range(n_conds)] * n_grps
    else:
        cond_lvls = cond_lvls * n_grps

    return pd.Series(cond_lvls, name='Condition')


def _define_vars(results, cond_lvls=None, grp_lvls=None):
    """
    Create a pandas data frame from `results` for easy plotting

    Uses the result dictionary returned by PLS as well as user-supplied
    condition and group label(s).

    Parameters
    ----------
    results : :obj:pyls.PLSResults
        The PLS result dictionary
    cond_lvls : list, optional
        List of condition labels
    grp_lvls : list, optional
        List of group labels

    Returns
    -------
    df : pd.DataFrame
        A pandas DataFrame with derived estimates (and upper- and lower-
        estimated error) for all latent variables
    """

    try:
        estimate = results.bootres.contrast
        ul = results.bootres.contrast_uplim
        ll = results.bootres.contrast_lolim
    except AttributeError:
        estimate = results.bootres.behavcorr
        ul = results.bootres.behavcorr_uplim
        ll = results.bootres.behavcorr_lolim

    n_grps = len(results.inputs.groups)
    n_conds = estimate.shape[1] // n_grps
    cond = _set_cond_lvls(n_conds, n_grps, cond_lvls=cond_lvls)
    grp = _set_group_lvls(n_conds, n_grps, grp_lvls=grp_lvls)

    num_est = estimate.shape[1] + 1  # for 1-based indexing in plots
    colnames = []
    for itm in ['Estimate LV', 'UL LV', 'LL LV']:
        for i in range(1, num_est):
            colnames.append(itm + str(i))

    df = pd.DataFrame(np.hstack((estimate, ul, ll)), columns=colnames)
    df = pd.concat([df, cond, grp], axis=1)
    return df


def _rearrange_df(df, plot_order):
    """
    Rearranged `df` according to `plot_order`

    In examining plots, users may wish to rearrange the order in which
    conditions are presented in order to ease visual interpretation. This
    function reorders the dataframe as desired

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing condition, group labels, and PLS results
    plot_order : list
        User-defined order in which to plot conditions

    Returns
    -------
    df : pd.DataFrame
        Provided dataframe `df` with re-ordered conditions
    """

    sorter_idx = dict(zip(plot_order, range(len(plot_order))))
    df['Cond_Arrange'] = df['Condition'].map(sorter_idx)
    df = df.sort_values(by=['Group', 'Cond_Arrange'], ascending=[False, True])
    return df.drop(columns=['Cond_Arrange'])
