# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns


def set_group_lvls(n_conds, n_grps, grp_lvls=None):
    """
    Derives a pandas data series of group labels for inclusion in
    the created data frame used in plotting. Uses group labels
    if provided by the user.

    Parameters
    ----------
    n_conds
        Number of conditions used in the analysis
    n_grps
        Number of groups used in the analysis
    grp_lvls
        (Optional) group labels

    Returns
    -------
    pd.Series
        A pandas series object of (optionally user-provided) group
          labels, aligned to the input data structure.

    """
    grping = []
    if grp_lvls is None:
        for i in range(n_grps):
            grping += ["Group" + str(i)] * n_conds
    else:
        for i in range(n_grps):
            grping.extend([grp_lvls[i]] * n_conds)
    return pd.Series(grping, name='Group')


def set_cond_lvls(n_conds, n_grps, cond_lvls=None):
    """
    Derives a pandas data series of condition labels for inclusion in
    the created data frame used in plotting. Uses condition labels
    if provided by the user.

    Parameters
    ----------
    n_rpt
        Number of times to repeat each group. This is usually
          equal to the number of conditions.
    n_grps
        Number of groups used in the analysis
    cond_lvls
        (Optional) condition labels

    Returns
    -------
    pd.Series
        A pandas series object of (optionally user-provided) condition
          labels, aligned to the input data structure.

    """
    if cond_lvls is None:
        cond_lvls = ["Condition" + str(i) for i in range(n_conds)] * n_grps
    else:
        cond_lvls = cond_lvls * n_grps

    return pd.Series(cond_lvls, name='Condition')


def define_vars(result_dict, cond_lvls=None, grp_lvls=None):
    """
    Define a pandas data frame for plotting. Uses the result dictionary
    returned by PLS, as well as user-supplied condition and group label(s),
    if supplied.

    Parameters
    ----------
    result_dict
        The PLS result dictionary
    cond_lvls
        (Optional) condition labels
    grp_lvls
        (Optional) group labels

    Returns
    -------
    pd.DataFrame
        A pandas series with derived estimates (and upper- and lower-
          estimated error) for all latent variables
    """
    estimate = result_dict['boot_result']['orig_usc']
    ul = result_dict['boot_result']['ulusc']
    ll = result_dict['boot_result']['llusc']

    n_grps = result_dict['num_subj_lst'].shape[1]
    n_conds = int(estimate.shape[1]/n_grps)
    cond = set_cond_lvls(n_conds, n_grps, cond_lvls=grp_lvls)
    grp = set_group_lvls(n_conds, n_grps, grp_lvls=grp_lvls)

    num_est = sum(estimate.shape[1]) + 1  # for 1-based indexing in plots
    colnames = []
    for itm in ['Estimate_LV', 'UL_LV', 'LL_LV']:
        for i in range(1, num_est):
            colnames.append(itm+str(i))

    df = pd.DataFrame(np.hstack((estimate, ul, ll)), columns=colnames)
    df = pd.concat([df, cond, grp], axis=1)
    return df


def rearrange_df(df, plot_order=None):
    """
    In examining plots, users may wish to rearrange the order
    in which conditions are presented in order to ease visual
    interpretation. This function reorders the dataframe as
    desired

    Parameters
    ----------
    df
        A pandas DataFrame containing condition, group labels and
          PLS results for latent variables
    (Optional) plot_order
        User-defined order in which to plot conditions

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with re-ordered conditions
    """
    sorter_idx = dict(zip(plot_order, range(len(plot_order))))
    df['Cond_Arrange'] = df['Condition'].map(sorter_idx)
    df = df.sort_values(by=['Group', 'Cond_Arrange'], ascending=[False, True])
    return df.drop(columns=['Cond_Arrange'])


def plot_bargraphs(df, idx):
    """
    Plots bargraphs for latent variables specified by ``idx``
    in a provided PLS results data frame

    Parameters
    ----------
    df
        A pandas DataFrame containing condition, group labels and
          PLS results for latent variables
    idx
        Index of desired latent variable to plot

    Returns
    -------
    matplotlib Axes obj
        A matplotlib axes object for saving or modifying
    """
    num_sig = (len(df.columns) - 2) // 3
    UL_num = idx + num_sig
    LL_num = idx + (num_sig * num_sig)
    ax = sns.barplot(x="Group", y=df[df.columns[idx]], hue="Condition", data=df,
                     capsize=0.1, errwidth=1.25, alpha=0.25, ci=None)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    x = [r.get_x() for r in ax.patches]
    nx = np.sort(x)
    abs_err = np.abs([df[df.columns[LL_num]].get_values(),
                      df[df.columns[UL_num]].get_values()] - df[df.columns[idx]].get_values())
    ax.errorbar(x=nx + (np.diff(nx).min() / 2),
                y=df[df.columns[idx]], fmt='none', yerr=abs_err)
    return ax
