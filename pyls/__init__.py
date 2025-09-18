# -*- coding: utf-8 -*-


__all__ = [
    "__version__"
]

from . import _version
__version__ = _version.get_versions()['version']


def __getattr__(name):
    """Lazy loading for all modules that require numpy."""
    if name == "examples":
        from . import examples
        return examples
    elif name == "behavioral_pls":
        from .types import behavioral_pls
        return behavioral_pls
    elif name == "meancentered_pls":
        from .types import meancentered_pls
        return meancentered_pls
    elif name == "pls_regression":
        from .types import pls_regression
        return pls_regression
    elif name == "load_results":
        from .io import load_results
        return load_results
    elif name == "save_results":
        from .io import save_results
        return save_results
    elif name == "matlab":
        from . import matlab
        return matlab
    elif name == "import_matlab_result":
        from .matlab import import_matlab_result
        return import_matlab_result
    elif name == "PLSInputs":
        from .structures import PLSInputs
        return PLSInputs
    elif name == "PLSResults":
        from .structures import PLSResults
        return PLSResults
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
