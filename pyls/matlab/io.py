"""Functions for importing PLS results from Matlab."""

from collections.abc import MutableMapping

import numpy as np
import scipy.io as sio

from ..structures import PLSResults

_result_mapping = (
    ("u", "x_weights"),
    ("s", "singvals"),
    ("v", "y_weights"),
    ("usc", "x_scores"),
    ("vsc", "y_scores"),
    ("lvcorrs", "y_loadings"),
    # permres
    ("perm_result_sprob", "pvals"),
    ("perm_result_permsamp", "permsamples"),
    # bootres
    ("boot_result_compare_u", "x_weights_normed"),
    ("boot_result_u_se", "x_weights_stderr"),
    ("boot_result_bootsamp", "bootsamples"),
    # splitres
    ("perm_splithalf_orig_ucorr", "ucorr"),
    ("perm_splithalf_orig_vcorr", "vcorr"),
    ("perm_splithalf_ucorr_prob", "ucorr_pvals"),
    ("perm_splithalf_vcorr_prob", "vcorr_pvals"),
    ("perm_splithalf_ucorr_ul", "ucorr_uplim"),
    ("perm_splithalf_vcorr_ul", "vcorr_lolim"),
    ("perm_splithalf_ucorr_ll", "ucorr_uplim"),
    ("perm_splithalf_vcorr_ll", "vcorr_lolim"),
    # inputs
    ("inputs_X", "X"),
    ("stacked_behavdata", "Y"),
    ("num_subj_lst", "groups"),
    ("num_conditions", "n_cond"),
    ("perm_result_num_perm", "n_perm"),
    ("boot_result_num_boot", "n_boot"),
    ("perm_splithalf_num_split", "n_split"),
    ("boot_result_clim", "ci"),
    ("other_input_meancentering_type", "mean_centering"),
    ("method", "method"),
)

_mean_centered_mapping = (
    ("boot_result_orig_usc", "contrast"),
    ("boot_result_distrib", "contrast_boot"),
    ("boot_result_ulusc", "contrast_ci_up"),
    ("boot_result_llusc", "contrast_ci_lo"),
)

_behavioral_mapping = (
    ("boot_result_orig_corr", "y_loadings"),
    ("boot_result_distrib", "y_loadings_boot"),
    ("boot_result_ulcorr", "y_loadings_ci_up"),
    ("boot_result_llcorr", "y_loadings_ci_lo"),
)


def _coerce_void(value):
    """
    Convert `value` to `value.dtype`.

    Parameters
    ----------
    value : array_like

    Returns
    -------
    value : dtype
        `Value` coerced to `dtype`
    """
    if np.squeeze(value).ndim == 0:
        return value.dtype.type(value.squeeze())
    else:
        return np.squeeze(value)


def _flatten(d, parent_key="", sep="_"):
    """
    Flatten nested dictionary `d` into single dictionary with new keyset.

    Parameters
    ----------
    d : dict
        Dictionary to be flattened
    parent_key : str, optional
        Key of parent dictionary of `d`. Default: ''
    sep : str, optional
        How to join keys of `d` with `parent_key`, if provided. Default: '_'

    Returns
    -------
    flat : dict
        Flattened input dictionary `d`

    Notes
    -----
    Taken directly from https://stackoverflow.com/a/6027615
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _rename_keys(d, mapping):
    """
    Rename keys in dictionary `d` based on tuples in `mapping`.

    Parameters
    ----------
    d : dict
        Dictionary with keys to be renamed
    mapping : list of tuples
        List of (oldkey, newkey) pairs to rename entries in `d`

    Returns
    -------
    renamed : dict
        Input dictionary `d` with keys renamed
    """
    new_dict = d.copy()
    for oldkey, newkey in mapping:
        try:
            new_dict[newkey] = new_dict.pop(oldkey)
        except KeyError:
            pass

    return new_dict


def import_matlab_result(fname, datamat="datamat_lst"):
    """
    Import `fname` PLS result from Matlab.

    Parameters
    ----------
    fname : str
        Filepath to output mat file obtained from Matlab PLS toolbox. Should
        contain at least a result struct object.
    datamat : str, optional
        Variable name of datamat ('X' array) provided to original PLS if it
        exists `fname`. By default the datamat is not stored in the PLS results
        structure, but if it is was saved in `fname` it can be loaded and
        cached in the returned results object. Default: 'datamat_lst'

    Returns
    -------
    results : :obj:`~.structures.PLSResults`
        Matlab results in a Python-friendly format
    """

    def get_labels(fields):
        labels = [k for k, v in sorted(fields.items(), key=lambda x: x[-1][-1])]
        return labels

    # load mat file using scipy.io
    matfile = sio.loadmat(fname)

    # if 'result' key is missing then consider this a malformed PLS result mat
    try:
        result = matfile.get("result")[0, 0]
    except (IndexError, TypeError):
        raise ValueError("Cannot get result struct from provided mat file") from None

    # convert result structure to a dictionary using dtypes as keys
    labels = get_labels(result.dtype.fields)
    result = {labels[n]: value for n, value in enumerate(result)}

    # convert sub-structures to dictionaries using dtypes as keys
    struct = ["boot_result", "perm_result", "perm_splithalf", "other_input"]
    for attr in struct:
        if result.get(attr) is not None:
            labels = get_labels(result[attr].dtype.fields)
            result[attr] = {
                labels[n]: _coerce_void(value)
                for n, value in enumerate(result[attr][0, 0])
            }

    # get input data from results file, if it exists
    X = matfile.get(datamat)
    result["inputs"] = dict(X=np.vstack(X[:, 0])) if X is not None else dict()

    # squeeze all the values so they're a bit more interpretable
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            result[key] = _coerce_void(val)

    # flatten the dictionary and rename the keys according to our mapping
    result = _rename_keys(_flatten(result), _result_mapping)
    if result["method"] == 3:
        result = _rename_keys(result, _behavioral_mapping)
        if "y_loadings_ci_up" in result:
            result["y_loadings_ci"] = np.stack(
                [result["y_loadings_ci_lo"], result["y_loadings_ci_up"]], axis=-1
            )
    else:
        result = _rename_keys(result, _mean_centered_mapping)
        if "contrast_ci_up" in result:
            result["contrast_ci"] = np.stack(
                [result["contrast_ci_lo"], result["contrast_ci_up"]], axis=-1
            )

    # index arrays - 1 to account for Matlab vs Python 1- vs 0-indexing
    for key in ["bootsamples", "permsamples"]:
        try:
            result[key] -= 1
        except KeyError:
            continue

    if result.get("n_split", None) is None:
        result["n_split"] = None

    # pack it into a `PLSResults` class instance for easy attribute access
    results = PLSResults(**result)

    return results
