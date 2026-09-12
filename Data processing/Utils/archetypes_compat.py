# =============================================================================
# archetypes_compat.py — sklearn/numpy compatibility shims for archetypes==0.6.2
# =============================================================================
# archetypes==0.6.2 (the only build installable on this machine without a
# C/C++ toolchain) predates two later removals: sklearn's private
# BaseEstimator._validate_data, and numpy's np.mat (removed in NumPy 2.0).
# Import this module BEFORE `from archetypes import AA` in any consumer —
# it patches sklearn.base.BaseEstimator and numpy at import time.
# =============================================================================

import numpy as np
import sklearn.base
from sklearn.utils.validation import validate_data as _validate_data_fn

if not hasattr(sklearn.base.BaseEstimator, "_validate_data"):
    def _validate_data(self, X, *args, **kwargs):
        return _validate_data_fn(self, X, *args, **kwargs)
    sklearn.base.BaseEstimator._validate_data = _validate_data

if not hasattr(np, "mat"):
    np.mat = np.asmatrix
