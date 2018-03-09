import numpy as np
import pandas as pd
import seaborn as sns


def set_group_lvls(n_rpt, grp_lvls, n_grps):
    """
    Parameters
    ----------

    Returns
    -------
    """
    Group = []
    for i in range(0, n_grps):
        Group.extend([grp_lvls[i]]*n_rpt)
    Group = pd.Series(Group, name='Group')
    return Group


def define_vars(result_dict, cond_lvls, grp_lvls):
    """
    Parameters
    ----------

    Returns
    -------
    """
    estimate = result_dict['boot_result']['orig_usc']
    ul = result_dict['boot_result']['ulusc']
    ll = result_dict['boot_result']['llusc']
    n_grps = result_dict['num_subj_lst'].shape[1]
    signif = result_dict['perm_result']['sprob']

    mask = signif < 0.05
    sigLV = mask.T[0]
    sigEstimate = estimate[sigLV].T
    sigUL = ul[sigLV].T
    sigLL = ll[sigLV].T
    n_rpt = int(estimate.shape[1]/n_grps)

    grp = set_group_lvls(n_rpt, grp_lvls, n_grps)
    cond = pd.Series(cond_lvls * n_grps, name='Condition')

    num_sig = sum(sigLV) + 1  # for 1-based indexing in plots
    colnames = []
    for itm in ['Estimate_LV', 'UL_LV', 'LL_LV']:
        for i in range(1, num_sig):
            colnames.append(itm+str(i))

    df = pd.DataFrame(np.hstack((sigEstimate, sigUL, sigLL)), columns=colnames)
    df = pd.concat([df, cond, grp], axis=1)

    return df


def rearrange_df(df, plot_order):
    """
    Parameters
    ----------

    Returns
    -------
    """
    sorter_idx = dict(zip(plot_order, range(len(plot_order))))
    df['Cond_Arrange'] = df['Condition'].map(sorter_idx)
    df = df.sort_values(by=['Group', 'Cond_Arrange'], ascending=[False, True])
    return df.drop(columns=['Cond_Arrange'])


def plot(df, idx, num_sig):
    """
    Parameters
    ----------

    Returns
    -------
    """
    UL_num = idx + num_sig
    LL_num = idx + (num_sig * num_sig)
    ax = sns.barplot(x="Group", y=df[df.columns[idx]], hue="Condition", data=df,
                     capsize=0.1, errwidth=1.25, alpha=0.25, ci=None)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    x = [r.get_x() for r in ax.patches]
    nx = np.sort(x)
    ax.errorbar(x=nx + (np.diff(nx).min() / 2),
                y=df[df.columns[idx]], fmt='none',
                yerr=np.abs(df[[df[df.columns[UL_num]],
                                df[df.columns[LL_num]]]].get_values().T - df.Estimate_LV1.get_values()))
    return ax
