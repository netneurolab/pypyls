# -*- coding: utf-8 -*-

import scipy.io as sio


def import_mean_centered_result(fname):
    """
    Imports ``fname`` PLS result from matlab

    Parameters
    ----------
    fname : str
        Filepath to result.mat file output by Matlab PLS

    Returns
    -------
    result : dict
        Matlab PLS result dictionary
    """

    result = sio.loadmat(fname)['result'][0, 0]
    all_labels = ['method', 'is_struct', 'u', 's', 'v', 'usc', 'vsc',
                  'num_subj_lst', 'num_conditions', 'perm_result',
                  'boot_result', 'other_input', 'field_descrip']
    perm_labels = ['num_perm', 'sp', 'sprob', 'permsamp', 'is_perm_splithalf']
    boot_labels = ['num_boot', 'clim', 'num_LowVariability_behav_boots',
                   'boot_type', 'nonrotated_boot', 'usc2', 'orig_usc', 'ulusc',
                   'llusc', 'ulusc_adj', 'llusc_adj', 'prop', 'distrib',
                   'bootsamp', 'compare_u', 'u_se', 'zero_u_se']
    other_labels = ['meancentering_type', 'corrmode']

    result = {all_labels[n]: value for n, value in enumerate(result)}
    result['perm_result'] = {perm_labels[n]: value for n, value in
                             enumerate(result['perm_result'][0, 0])}
    result['boot_result'] = {boot_labels[n]: value for n, value in
                             enumerate(result['boot_result'][0, 0])}
    result['other_input'] = {other_labels[n]: value for n, value in
                             enumerate(result['other_input'][0, 0])}

    return result
