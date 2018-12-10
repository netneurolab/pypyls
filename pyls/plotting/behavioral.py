# -*- coding: utf-8 -*-
"""
Functions for plotting results from a behavioral PLS
"""

from .meancentered import plot_contrast


def plot_behaviors(results, lv=0, behaviors=None, cond_labels=None,
                   group_labels=None, cond_order=None, error_kws=None,
                   legend=True, **kwargs):
    """
    Plots group / condition contrast from `results` for a provided `lv`

    Parameters
    ----------
    results : :obj:pyls.PLSResults
        The PLS result dictionary
    lv : int, optional
        Index of desired latent variable to plot. Uses zero-indexing, so the
        first latent variable is `lv=0`. Default: 0
    behaviors : list, optional
        Labels for behaviors (i.e., columns in `Y` matrix provided to original
        PLS). If not specified, will check if `Y` matrix was a pandas.DataFrame
        and use columns labels; if not, will default to "Behavior X". Default:
        None
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

    ax = plot_contrast(results, lv=lv, cond_labels=cond_labels,
                       group_labels=group_labels, cond_order=cond_order,
                       error_kws=error_kws, legend=legend, **kwargs)

    if behaviors is not None:
        ax.set_xlabels(behaviors)

    return ax
