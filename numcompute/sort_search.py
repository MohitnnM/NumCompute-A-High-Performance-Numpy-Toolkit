"""
Implements the sorting, selection and searching algorithms for NumCompute.

This module includes both stable and multi-key sorting, top-k selection, 
a manual quickselect implementation, and a binary search for locating 
values or finding their insertion position in sorted arrays. 
"""
import numpy as np

# SORTING
def stable_sort(values):

    sorted_array = np.sort(values, kind="stable")

    return sorted_array

def multi_key_sort(self):
    pass

#SELECTION
def topk(values, k, largest=True, return_indices=True):
    """
        Selects the k largets or smallest values from the given array.

        Parameters:
            values: The given NumPy array, containing values to select from.
            k: The number of values to select.
            largest: Boolean to specify whether to select the largest, if 
            True, or the smallest, if False, k values. 
            return_indices: Boolean to specify if the function returns an
                array of indices of selected values, if True, or values
                themselves, if False. 
        
        Returns:
            A NumPy array containing either the selected values or their
                indices.
        
        Raises:
            ValueError: If k is less than 1, equal to 0, or greater than
                the number of values.

        Time Complexity:
            O(n + k log(k)), where n is the number of input values, and
            k if the number of selected values.

        Space Complexity:
            O(n), mainly as np.argpartition returns index array, proportional
            to input size.
    """
    if k <= 0 or k > len(values):
        raise ValueError(
            "k must be greater than 0 and less than the number of values."
        )

    
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

def quickselect(self):
    pass

#SEARCHING
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

        Time complexity:
            Best case: O(1)
            Average and worst case: O(log n)

        Space Complexity:
            O(1)
    """

    left = 0
    right = len(sorted_array) - 1
    

    while left <= right:
        mid_idx = (left+right) // 2 

        if sorted_array[mid_idx] == x:
            insertion_idx = mid_idx
            return insertion_idx, True
        
        if  x < sorted_array[mid_idx]:
            right = mid_idx - 1

        if x > sorted_array[mid_idx] :
            left = mid_idx + 1

    return left, False

print(topk(np.array([2,10,3,24,62,15,1,50,4,7]),3,False, False))