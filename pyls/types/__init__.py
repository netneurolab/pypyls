# -*- coding: utf-8 -*-
"""The primary PLS decomposition methods for use in conducting PLS analyses."""

__all__ = ["behavioral_pls", "meancentered_pls", "pls_regression"]

from .behavioral import behavioral_pls
from .meancentered import meancentered_pls
from .regression import pls_regression
