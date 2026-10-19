"""
Contains shared numerical utility functions used by other modules
throughout the NumCompute package.

This is mainly created to house reusable helper functions that support
the modules in the NumCompute package, including functions such as
distance calculation and activation functions. It may also include
numerically stable operations like log-sum-exp, batching and shared
top-k utilities.

The purpose of this module is to provide the NumCompute packages with
common numerical operations that do not belong exclusively to one
module. Where applicable, the implementation will use NumPy
vectorisation and numerical stability.
"""

import numpy as np

from numcompute.exceptions import NotFittedError


def check_array(X, dtype=float, allow_nan=False):

    """
        Converts input to a NumPy array and checks that it is a non-empty
        2D array.

        Parameters:
            X:
                Array-like input of shape (n_samples, n_features).

            dtype:
                Target dtype. Defaults to float. Use None to keep the
                input's dtype (e.g. strings for OneHotEncoder).

            allow_nan:
                If False (default), numeric input containing NaN or Inf
                is rejected.

        Returns:
            A 2D NumPy array of shape (n_samples, n_features).

        Raises:
            TypeError:
                If X cannot be converted to dtype.

            ValueError:
                If X is ragged, not 2D, empty, or contains NaN/Inf when
                allow_nan is False. 1D input is rejected, not reshaped.

        Time Complexity:
            O(n) where n is the number of elements in X.

        Space Complexity:
            O(n) if X has to be copied or converted, otherwise O(1).
    """

    try:
        X_arr = np.asarray(X, dtype=dtype)
    except (TypeError, ValueError) as e:
        if "inhomogeneous" in str(e):
            raise ValueError(
                "Input forms a ragged array with inconsistent row lengths."
            ) from e
        dtype_name = getattr(dtype, "__name__", dtype)
        raise TypeError(f"Input cannot be converted to {dtype_name}.") from e

    if X_arr.ndim == 0:
        raise ValueError("Expected a 2D array, got a scalar instead.")
    if X_arr.ndim == 1:
        raise ValueError(
            "Expected a 2D array, got a 1D array instead. Use "
            "X.reshape(-1, 1) for a single feature or X.reshape(1, -1) "
            "for a single sample."
        )
    if X_arr.ndim > 2:
        raise ValueError(
            f"Expected a 2D array, got a {X_arr.ndim}D array instead."
        )

    if X_arr.size == 0:
        raise ValueError("Input array is empty (zero rows or columns).")

    # Only numeric arrays can hold NaN/Inf, so string arrays skip this.
    if not allow_nan and np.issubdtype(X_arr.dtype, np.number):
        if not np.all(np.isfinite(X_arr)):
            raise ValueError(
                "Input contains NaN or Inf. Set allow_nan=True to allow them."
            )

    return X_arr


def check_is_fitted(estimator, attributes):

    """
        Checks that an estimator has been fitted.

        Fitted attributes (e.g. mean_, scale_) must only be created inside
        fit, never in __init__, so their absence means "not fitted".

        Parameters:
            estimator:
                The transformer or model instance to check.

            attributes:
                Name, or list of names, of attributes set by fit.

        Raises:
            NotFittedError:
                If any of the attributes is missing.

        Time Complexity:
            O(k) where k is the number of attributes.

        Space Complexity:
            O(1).
    """

    if isinstance(attributes, str):
        attributes = [attributes]

    if not all(hasattr(estimator, attr) for attr in attributes):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' before using this estimator."
        )

def check_1d_array(X, dtype=float, allow_nan=False, allow_empty_arr=False):
    """
        Converts input to a NumPy array and checks that it is a non-empty
        1D array.

        Parameters:
            X:
                Array-like input of shape (n_features,).

            dtype:
                Target dtype. Defaults to float. Use None to keep the
                input's dtype (e.g. strings for OneHotEncoder).

            allow_nan:
                If False (default), numeric input containing NaN or Inf
                is rejected.

        Returns:
            A 1D NumPy array of shape (n_features,).

        
        Raises:
            TypeError:
                If X cannot be converted to dtype.

            ValueError:
                If X is ragged, not 1D, empty, or contains NaN/Inf when
                allow_nan is False.

        Time Complexity:
            O(n) where n is the number of elements in X.

        Space Complexity:
            O(n) if X has to be copied or converted, otherwise O(1).
    """

    try:
        X_arr = np.asarray(X, dtype=dtype)
    except (TypeError, ValueError) as e:
        if "inhomogeneous" in str(e):
            raise ValueError(
                "Input forms a ragged array with inconsistent row lengths."
            ) from e

        dtype_name = getattr(dtype, "__name__", dtype)
        raise TypeError(
            f"Input cannot be converted to {dtype_name}."
        ) from e

    if X_arr.ndim != 1:
        raise ValueError(
            f"Expected a 1D array, got a {X_arr.ndim}D array instead."
        )

    if X_arr.size == 0 and not allow_empty_arr:
        raise ValueError("Input array is empty.")

    # Only numeric arrays can hold NaN/Inf, so string arrays skip this.
    if not allow_nan and np.issubdtype(X_arr.dtype, np.number):
        if not np.all(np.isfinite(X_arr)):
            raise ValueError(
                "Input contains NaN or Inf. Set allow_nan=True to allow them."
            )

    return X_arr