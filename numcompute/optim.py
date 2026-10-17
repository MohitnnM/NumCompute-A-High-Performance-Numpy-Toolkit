"""
Finite-difference gradient and Jacobian calculations for NumCompute.

This module implements numerical methods to estimate the derivatives using
small changes to the input values.

It also supports forward and central difference methods to calculate the
gradients, as well as Jacobian matrices for functions with multiple outputs.

Important to note that the step size for numerical approximation can be
adjusted, and that all and any calculations are designed to function with
NumPy arrays.
"""

import numpy as np


def _check_inputs(x, h, method):
    """
    Converts x to a float array and checks x, h and method.

    Raises:
        ValueError:
            If x is not 0D or 1D, h is not positive, or method is not
            "forward" or "central".
    """

    x_arr = np.asarray(x, dtype=float)

    if x_arr.ndim > 1:
        raise ValueError(f"x must be a scalar or 1D vector, got {x_arr.ndim}D.")
    if method not in ("forward", "central"):
        raise ValueError(f"method must be 'forward' or 'central', got {method!r}.")
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}.")

    return x_arr


def _finite_difference_loop(func, x_arr, base_val, h, method):
    """
    Returns a list with one derivative estimate per input dimension.

    Nudges one input at a time by h. The loop is over input dimensions,
    not data, because func only accepts one point at a time.
    """

    x_1d = np.atleast_1d(x_arr)
    n = len(x_1d)

    nudges = h * np.eye(n)
    derivatives = []

    for i in range(n):
        nudge_pos = (x_1d + nudges[i]).reshape(x_arr.shape)

        if method == "forward":
            diff = (func(nudge_pos) - base_val) / h
        else:
            nudge_neg = (x_1d - nudges[i]).reshape(x_arr.shape)
            diff = (func(nudge_pos) - func(nudge_neg)) / (2 * h)

        derivatives.append(diff)

    return derivatives


def grad(f, x, h=1e-5, method="central"):
    """
    Estimates the gradient of a scalar function f at point x.

    Forward: (f(x + h) - f(x)) / h, error O(h).
    Central: (f(x + h) - f(x - h)) / (2h), error O(h^2).

    Parameters:
        f:
            Function taking an array like x and returning one number.

        x:
            Point to evaluate at, a scalar or 1D array-like.

        h:
            Step size. Defaults to 1e-5.

        method:
            "central" (default) or "forward".

    Returns:
        Array with the same shape as x.

    Raises:
        ValueError:
            If x, h or method are invalid, or f(x) is not a single number.

    Time Complexity:
        O(n) calls to f for n inputs (2n for central).

    Space Complexity:
        O(n^2) for the nudge matrix.
    """

    x_arr = _check_inputs(x, h, method)
    base_val = f(x_arr)

    if np.ndim(base_val) != 0:
        raise ValueError("f(x) must return a single number for grad.")

    derivatives = _finite_difference_loop(f, x_arr, base_val, h, method)

    return np.array(derivatives).reshape(x_arr.shape)


def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimates the Jacobian matrix of a vector function F at point x.

    J[i, j] is the derivative of output i with respect to input j, so
    nudging input j gives column j.

    Parameters:
        F:
            Function taking an array like x and returning m numbers.

        x:
            Point to evaluate at, a scalar or 1D array-like of length n.

        h:
            Step size. Defaults to 1e-5.

        method:
            "central" (default) or "forward".

    Returns:
        Array of shape (m, n).

    Raises:
        ValueError:
            If x, h or method are invalid.

    Time Complexity:
        O(n) calls to F for n inputs (2n for central).

    Space Complexity:
        O(n^2 + m * n) for the nudge matrix and the result.
    """

    x_arr = _check_inputs(x, h, method)
    base_val = F(x_arr)

    derivatives = _finite_difference_loop(F, x_arr, base_val, h, method)

    return np.column_stack(derivatives)
