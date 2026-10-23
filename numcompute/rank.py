"""
Provides ranking and percentile operations for numerical data.

This module computes the rankings and the percentiles. The rankings are
used to preserve the relationship to the original observations, handling
ties by assigning tied values their average rank, the same dense rank,
or distinct ordinal ranks based on original order.

Percentile is used to identify the value that corresponds to a given position
within an ordered dataset. If the given position falls between two data
points, then the result is determined using linear, lower, higher, or
midpoint interpolations.
"""
import numpy as np

from numcompute.utils import check_1d_array

RANK_METHODS = ("average", "ordinal", "dense")
INTERPOLATIONS = ("linear", "lower", "higher", "midpoint")


def rank(data, method="average"):
    """
        Ranks numerical data from 1 to n datapoints.

        Parameters:
            data:
                The numerical data to be ranked, handles tied values.
                Multi-dimensional data is flattened first.

            method:
                The ranking method to be performed if values are tied.
                Defaults to "average":
                - average: tied values receive the average of the
                positions they occupy in sorted order.
                - ordinal: tied values are ranked by their original
                order. All items receive distinct ranks.
                - dense: tied values share a rank, and the next distinct
                value receives the next rank, with no gaps.

        Returns:
            A 1D floating-point NumPy array with shape (n,), where n is
            the number of elements in data.
            Missing values are represented using np.nan, and are left
            out when ranking the remaining values.

        Raises:
            TypeError:
                If data provided does not have numerical values to
                rank.

            ValueError:
                If data is empty, or if a method which does not exist
                for ranking is inputted.

        Time Complexity:
            O(n log n), where n is the number of elements, as every
            method sorts the data. The tie handling after the sort is
            O(n).

        Space Complexity:
            O(n) for the sorted, index and rank arrays.
    """
    if method not in RANK_METHODS:
        raise ValueError(
            f"Unknown ranking method '{method}'. Use one of {RANK_METHODS}."
        )

    data_flattened = check_1d_array(np.ravel(data), allow_nan=True)

    ranks = np.full(data_flattened.shape, np.nan)
    is_present = ~np.isnan(data_flattened)
    values = data_flattened[is_present]

    match method:
        case "average":
            _, inverse, counts = np.unique(
                values, return_inverse=True, return_counts=True
            )

            last_position = np.cumsum(counts)
            first_position = last_position - counts + 1
            average_rank = (first_position + last_position) / 2

            ranks[is_present] = average_rank[inverse]

        case "ordinal":
            sorted_idx = np.argsort(values, kind="stable")
            ranks[is_present] = np.argsort(sorted_idx) + 1

        case "dense":
            _, inverse = np.unique(values, return_inverse=True)
            ranks[is_present] = inverse + 1
    return ranks


def percentile(data, q, interpolation="linear"):
    """
        Calculates the percentile of a dataset, for a provided q value.

        Parameters:
            data:
                Numerical data for which to perform a percentile
                computation. Does not need to be sorted, and
                multi-dimensional data is flattened first.

            q:
                The provided percentile, in decimals between 0 and 1
                (e.g. 0.25 for the 25th percentile).

            interpolation:
                The kind of percentile which is computed when q falls
                between two data points. Has the following interpolations:
                - linear: the default, using linear interpolation formula.
                - lower: lower data point.
                - higher: higher data point.
                - midpoint: the average between lower and higher point.

        Returns:
            Percentile value as a float.
            Returns np.nan if data contains missing values.

        Raises:
            TypeError:
                If data provided does not have numerical values to
                compute a percentile with, or if q is not a number.

            ValueError:
                If data is empty, q is outside [0, 1], or an unknown
                interpolation is inputted.

        Time Complexity:
            O(n log n), where n is the number of elements, from sorting
            the data. Finding the indices and interpolating is O(1).

        Space Complexity:
            O(n) for the sorted copy of the data.
    """
    if interpolation not in INTERPOLATIONS:
        raise ValueError(
            f"Unknown interpolation '{interpolation}'. "
            f"Use one of {INTERPOLATIONS}."
        )
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1.")

    data_flattened = check_1d_array(np.ravel(data), allow_nan=True)
    if np.isnan(data_flattened).any():
        return np.nan

    data_sorted = np.sort(data_flattened)

    virtual_idx = q * (len(data_sorted) - 1)
    lower_idx = int(np.floor(virtual_idx))
    higher_idx = int(np.ceil(virtual_idx))
    fractional = virtual_idx - lower_idx

    a = data_sorted[lower_idx]
    b = data_sorted[higher_idx]

    match interpolation:
        case "linear":
            result = a + (b - a) * fractional
        case "lower":
            result = a
        case "higher":
            result = b
        case "midpoint":
            result = (a + b) / 2

    return float(result)

arr = np.array([0, 2, 3, 2, 7, 2])
rank(arr)
