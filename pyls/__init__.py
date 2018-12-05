# -*- coding: utf-8 -*-

__all__ = [
    '__author__', '__description__', '__email__', '__license__',
    '__maintainer__', '__packagename__', '__url__', '__version__',
    'behavioral_pls', 'meancentered_pls', 'import_matlab_result',
    'PLSResults', 'save_results', 'load_results'
]

from .info import (
    __author__,
    __description__,
    __email__,
    __license__,
    __maintainer__,
    __packagename__,
    __url__,
    __version__
)

try:
    from .io import load_results, save_results
    from .matlab import import_matlab_result
    from .structures import PLSResults
    from .types import (behavioral_pls, meancentered_pls)
except ImportError:
    pass
