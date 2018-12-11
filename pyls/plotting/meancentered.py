# -*- coding: utf-8 -*-
"""
Functions for plotting results from a mean-centered PLS
"""

import numpy as np
import seaborn as sns
from .utils import _define_vars, _rearrange_df


def plot_contrast(results, lv=0, cond_labels=None, group_labels=None,
                  cond_order=None, error_kws=None, legend=True, **kwargs):
    """
    Plots group / condition contrast from `results` for a provided `lv`

    Parameters
    ----------
    results : :obj:pyls.PLSResults
        The PLS result dictionary
    lv : int, optional
        Index of desired latent variable to plot. Uses zero-indexing, so the
        first latent variable is `lv=0`. Default: 0
    cond_labels : list, optional
        List of condition labels as they were supplied to the original PLS.
        If not supplied, uses "Condition X" as label. Default: None
    group_labels : list, optional
        List of group labels as they were supplied to the original PLS. If
        not supplied, uses "Group X" as label. Default: None
    cond_order : list, optional
        Desired order for plotting conditions. If not supplied, plots
        conditions in order they were provided to original PLS. Default: None
    error_kws : dict, optional
        Dictionary supplying keyword arguments for errorbar plotting. Default:
        None
    legend : bool, optional
        Whether to plot legend automatically. Default: True
    **kwargs : key, value mappings
        Keywords arguments passed to :obj:seaborn.barplot

    Returns
    -------
    ax : matplotlib.axes.Axis
        A matplotlib axes object for saving or modifying
    """

    error_opts = dict(fmt='none', ecolor='black')
    if error_kws is not None:
        if not isinstance(error_kws, dict):
            raise TypeError('Provided error_kws must be a dictionary, not a '
                            '{}. Please check inputs and try again.'
                            .format(type(error_kws)))
        error_opts.update(**error_kws)

    df = _define_vars(results, cond_lvls=cond_labels, grp_lvls=group_labels)
    if cond_order is not None:
        diff_cond = set(cond_order) - set(cond_labels)
        if len(diff_cond) > 0:
            raise ValueError('Provided cond_order had labels not provided in '
                             'cond_labels: {}'.format(list(diff_cond)))
        df = _rearrange_df(df, cond_order)

    num_sig = (len(df.columns) - 2) // 3
    barplot_opts = dict(capsize=0.1, errwidth=1.25, alpha=0.25, ci=None)
    if len(kwargs) > 0:
        barplot_opts.update(**kwargs)
    ax = sns.barplot(x='Group', y=df[df.columns[lv]], hue='Condition',
                     data=df, **barplot_opts)
    if legend:
        ax.legend(bbox_to_anchor=(1.1, 1.05))
    else:
        ax.legend_.set_visible(False)

    xbar = np.array([r.get_x() for r in ax.patches])
    width = np.array([r.get_width() for r in ax.patches])[xbar.argsort()]
    xbar = xbar[xbar.argsort()]
    abs_err = np.abs([df[df.columns[lv + (num_sig * 2)]].get_values(),
                      df[df.columns[lv + num_sig]].get_values()]
                     - df[df.columns[lv]].get_values())
    ax.errorbar(x=xbar + (width / 2), y=df[df.columns[lv]], yerr=abs_err,
                **error_opts)

    return ax
