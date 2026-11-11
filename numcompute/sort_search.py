"""
Implements the sorting, selection and searching algorithms for NumCompute.

This module includes both stable and multi-key sorting, top-k selection,
a manual quickselect implementation, and a binary search for locating
values or finding their insertion position in sorted arrays.
"""

import numpy as np

from numcompute.utils import check_1d_array, check_array


# SORTING
def stable_sort(values):
    """
    Sorts a 1D array in ascending order, preserving relative order
    of elements, with equal comparison.

    Parameters:
        values:
            1D array-like containing values to sort.

    Returns:
        New 1D NumPy array containing sorted values.

    Raises:
        ValueError, if values are not 1 dimensional.

    Time Complexity:
        O(n log n), where n refers to the number of elements.

    Space Complexity:
        O(n) for the returned sorted array.
    """
    values = check_1d_array(values, dtype=None, allow_empty_arr=True, allow_nan=True)

    sorted_array = np.sort(values, kind="stable")

    return sorted_array


def multi_key_sort(values, keys_array):
    """
    Sorts the rows of the given 2D NumPy array using multiple columns
    as sorting keys. All keys are sorted in ascending order.

    Parameters:
        values: The given 2D NumPy array, containing values to sort.

        keys_array: A sequence of column indices that specify the
            sorting priority. The first index is the primary key,
            followed by any number of necessarry tie-breaking keys.



    Returns:
        A new 2D NumPy array of columns sorted according to specified
        sorting column keys.

    Raises:
        ValueError:
            If values is not a non-empty 2D array or if no
            sorting keys are provided.

        IndexError:
            If the sorting keys are out of bounds.

    Time Complexity:
        O(n log n), where n is the number of rows being sorted.

    Space Complexity:
        O(n * m), where n is the number of rows and m is the number
        of columns.
    """
    values = check_array(values, dtype=None)

    if len(keys_array) == 0:
        raise ValueError("Please provide at least one sorting key")

    for key in keys_array:
        if not (0 <= key < values.shape[1]):
            raise IndexError(
                f"The column index {key} is out of bounds for array with "
                f"{values.shape[1]} columns."
            )

    sort_keys = [values[:, key] for key in reversed(keys_array)]

    sort_indices = np.lexsort(sort_keys)
    sorted_values_arr = values[sort_indices]

    return sorted_values_arr


# SELECTION
def topk(values, k, largest=True, return_indices=True):
    """
    Selects the k largets or smallest values from the given array.

    Parameters:
        values:
            A non-empty NumPy array, containing values to select from.

        k:
            The number of values to select. Must be between 1 and
            number of input values, inclusive.

        largest:
            Boolean to specify whether to select the largest, if
            True, or the smallest, if False, k values. Defaults
            to True.

        return_indices: Boolean to specify if the function returns an
            array of indices of selected values, if True, or values
            themselves, if False. Defaults to True.

    Returns:
        A NumPy array containing either the selected values or their
        indices.

    Raises:
        ValueError:
            If array is empty, not one-dimensional, contains NaN or Inf,
            or k is outside of valid range.

    Time Complexity:
        O(n + k log(k)), where n is the number of input values, and
        k if the number of selected values.

    Space Complexity:
        O(n), mainly as np.argpartition returns index array, proportional
        to input size.
    """

    values = check_1d_array(values, dtype=None, allow_empty_arr=False, allow_nan=False)

    if k <= 0 or k > len(values):
        raise ValueError("k must be greater than 0 and less than the number of values.")

    if largest:
        boundary = len(values) - k

        partitioned_arr = np.argpartition(values, boundary)
        top_k_idx = partitioned_arr[-k:]
        top_k_vals = values[top_k_idx]

        sorted_k = np.argsort(top_k_vals)[::-1]

    else:
        boundary = k - 1

        partitioned_arr = np.argpartition(values, boundary)
        top_k_idx = partitioned_arr[:k]
        top_k_vals = values[top_k_idx]

        sorted_k = np.argsort(top_k_vals)

    sorted_idx = top_k_idx[sorted_k]

    if return_indices:
        return sorted_idx

    return values[sorted_idx]


def _partition(values, left, right):
    """
    Helper function partitions values[left:right + 1] into three regions:
        - values < pivot
        - values == pivot
        - values > pivot

    Parameters:
        values: A 1D array that is modified in place.

        left: The starting index of the partition array.

        right: The ending index of the partition array.

    Returns:
        A tuple that has (equal_start, equal_end), and gives the inclusive
        boundaries of the region that contains values equal to the pivot.

    Time Complexity:
        O(n)

    Space Complexity:
        O(1)
    """
    pivot = values[right]

    lower_idx = left
    current_idx = left
    upper_idx = right

    while current_idx <= upper_idx:
        if values[current_idx] < pivot:
            values[lower_idx], values[current_idx] = (
                values[current_idx],
                values[lower_idx],
            )
            lower_idx += 1
            current_idx += 1
            
        elif values[current_idx] > pivot:
            values[current_idx], values[upper_idx] = (
                values[upper_idx],
                values[current_idx],
            )

            upper_idx -= 1
        else:
            current_idx += 1
    
    return lower_idx, upper_idx


def quickselect(values, k):
    """
    Selects the k-th smallest element in an unordered 1D array using
    the Quickselect algorithm.

    value of k uses 1-based indexing, so k=1 selects the smallest
    element and k=len(values) or k equals the length of the given array
    returns the largest element. Partioning is performed in place.

    Parameters:
        values:
            The given NumPy array, containing values to select from.

        k:
            The rank of the desired element using 1-based indexing.

    Returns:
        k-th smallest value.

    Raises:
        ValueError:
            If the array is empty, not one dimensional, contains
            NaN or Inf, or k is outside valid range.

    Time Complexity:
        Average case: O(n)
        Worst case: O(n^2)

    Space Complexity:
        O(1)
    """
    values = check_1d_array(values, dtype=None, allow_empty_arr=False, allow_nan=False)

    if k <= 0 or k > len(values):
        raise ValueError("k must be between 1 and the number of values, inclusive.")

    target_idx = k -1

    left = 0
    right = len(values) - 1

    while left <= right:
        equal_start, equal_end = _partition(values, left, right)

        if target_idx < equal_start:
            right = equal_start - 1

        elif target_idx > equal_end:
            left = equal_end + 1

        else:
            return values[target_idx]


# SEARCHING
def binary_search(sorted_array, x):
    """
    Searches for the target value in the given array.

    Parameters:
        sorted_array: A NumPy array sorted in ascending order.

        x: The target value, i.e. the value being searched for.

    Return Values:
        A tuple that contains both the insertion_idx, the index of
        the target value if found and the existence boolean indicating
        if value exists in the array.

    Raises:
        ValueError:
            If input array, sorted_array, is not one-dimensional.

    Time complexity:
        Best case: O(1)
        Average and worst case: O(log n)

    Space Complexity:
        O(1)
    """
    sorted_array = check_1d_array(
        sorted_array, dtype=None, allow_empty_arr=True, allow_nan=False
    )

    left = 0
    right = len(sorted_array) - 1

    while left <= right:
        mid_idx = (left + right) // 2

        if sorted_array[mid_idx] == x:
            insertion_idx = mid_idx
            return insertion_idx, True

        if x < sorted_array[mid_idx]:
            right = mid_idx - 1

        if x > sorted_array[mid_idx]:
            left = mid_idx + 1

    return left, False
