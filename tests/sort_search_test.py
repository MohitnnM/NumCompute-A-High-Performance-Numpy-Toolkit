"""
    Unit tests for sort_search.py module
"""

import unittest

import numpy as np

import numcompute.sort_search as ss


class TestStableSort(unittest.TestCase):

    def test_unsorted_array(self):
        arr = np.array([4, 2, 6, 1, 5, 3])
        expected_arr = np.array([1, 2, 3, 4, 5, 6])
        
        result = ss.stable_sort(arr)

        np.testing.assert_array_equal(result, expected_arr)

    def test_sorted_array(self):
        arr = np.array([1, 2, 3, 4, 5, 6])
        expected_arr = np.array([1, 2, 3, 4, 5, 6])
        
        result = ss.stable_sort(arr)

        np.testing.assert_array_equal(result, expected_arr)

    def test_duplicate_vals(self):
        arr = np.array([4, 2, 6, 6, 4, 3])
        expected_arr = np.array([2, 3, 4, 4, 6, 6])
        
        result = ss.stable_sort(arr)

        np.testing.assert_array_equal(result, expected_arr)

    def test_if_array_empty(self):
        arr = np.array([])
        expected_arr = np.array([])

        result = ss.stable_sort(arr)
        
        np.testing.assert_array_equal(result, expected_arr)

    def test_negative_vals(self):
        arr = np.array([-4, 2, 5, -6, 1, 3])
        expected_arr = np.array([-6, -4, 1, 2, 3, 5])
        
        result = ss.stable_sort(arr)

        np.testing.assert_array_equal(result, expected_arr)

    def test_floating_vals(self):
        arr = np.array([4.2, 2.2, 2.3, 6.5, 1.2, 1.4, 1.6, 4.2, 2.6, 5])
        expected_arr = np.array([1.2, 1.4, 1.6, 2.2, 2.3, 2.6, 4.2, 4.2, 5, 6.5])
        
        result = ss.stable_sort(arr)

        np.testing.assert_array_equal(result, expected_arr)

    def test_non_contiguous_array(self):
        arr = np.array([6, 1, 2, 4, 5, 3])  
        sliced_arr = arr[::2]

        expected_arr = np.array([2, 5, 6])
        
        result = ss.stable_sort(sliced_arr)

        np.testing.assert_array_equal(result, expected_arr)

        


class TestMultiKeySort(unittest.TestCase):

    def test_multi_sort_with_2_keys(self):
        arr = np.array([
            [80, 22],
            [90, 24],
            [80, 20],
            [90, 21]
        ])

        expected_arr = np.array([
            [80, 20],
            [80, 22],
            [90, 21],
            [90, 24]
        ])

        result = ss.multi_key_sort(arr, [0, 1])

        np.testing.assert_array_equal(result, expected_arr)

    def test_multi_sort_with_multiple_keys(self):
        arr = np.array([
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 5],
            [1, 1, 2],
            [2, 2, 1],
            [2, 1, 1], 
            [2, 2, 5]
        ])

        expected_arr = np.array([
            [1, 1, 2],
            [1, 2, 3],
            [1, 2, 4],
            [1, 3, 5],
            [2, 1, 1],
            [2, 2, 1],
            [2, 2, 5]
        ])

        result = ss.multi_key_sort(arr, [0, 1, 2])

        np.testing.assert_array_equal(result, expected_arr)

    def test_one_sorting_key(self):
        
        arr = np.array([
            [2, 2],
            [4, 3],
            [1, 4],
            [3, 5]
        ])

        expected_arr = np.array([
            [1, 4],
            [2, 2],
            [3, 5],
            [4, 3]
        ])

        result = ss.multi_key_sort(arr, [0])

        np.testing.assert_array_equal(result, expected_arr)

    def test_raise_value_error_if_array_empty(self):
        arr = np.array([])

        with self.assertRaises(ValueError):
            ss.multi_key_sort(arr, [])

    def test_raise_value_error_if_keys_arr_empty(self):
        arr = np.array([
            [2, 2],
            [4, 3],
            [1, 4],
            [3, 5]
        ])

        with self.assertRaises(ValueError):
            ss.multi_key_sort(arr, [])
    
    def test_raise__index_error_if_keys_negative(self):
        arr = np.array([
            [2, 2],
            [4, 3],
            [1, 4],
            [3, 5]
        ])

        with self.assertRaises(IndexError):
            ss.multi_key_sort(arr, [-1])

    def test_raise_index_error_if_keys_out_of_range(self):
        arr = np.array([
            [2, 2],
            [4, 3],
            [1, 4],
            [3, 5]
        ])

        with self.assertRaises(IndexError):
            ss.multi_key_sort(arr, [2])

    def test_raise_value_error_if_array_not_two_d(self):
        arr = np.array([2, 3, 4, 5])

        with self.assertRaises(ValueError):
            ss.multi_key_sort(arr, [0])


class TestTopK(unittest.TestCase):
    def test_largest_k_values(self):

        arr = np.array([5, 4, 10, 23, 19, 1, 6, 15, 26])

        expected_arr = np.array([26, 23, 19])

        result = ss.topk(arr, 3, True, False)

        np.testing.assert_array_equal(result, expected_arr)

    def test_smallest_k_values(self):

        arr = np.array([5, 4, 10, 23, 19, 1, 6, 15, 26])

        expected_arr = np.array([1, 4, 5])

        result = ss.topk(arr, 3, False, False)

        np.testing.assert_array_equal(result, expected_arr)

    def test_largest_k_indices(self):

        arr = np.array([5, 4, 10, 23, 19, 1, 6, 15, 26])

        expected_arr = np.array([8, 3, 4])

        result = ss.topk(arr, 3, True, True)

        np.testing.assert_array_equal(result, expected_arr)

    def test_smallest_k_indices(self):

        arr = np.array([5, 4, 10, 23, 19, 1, 6, 15, 26])

        expected_arr = np.array([5, 1, 0])

        result = ss.topk(arr, 3, False, True)

        np.testing.assert_array_equal(result, expected_arr)

    def test_smallest_k_value(self):

        arr = np.array([5, 4, 10, 23, 19, 1, 6, 15, 26])

        expected_arr = np.array([26])

        result = ss.topk(arr, 1, True, False)

        np.testing.assert_array_equal(result, expected_arr)

    def test_largest_k_value(self):

        arr = np.array([5, 4, 10, 23, 19, 1, 6, 15, 26])

        expected_arr = np.array([26, 23, 19, 15, 10, 6, 5, 4, 1])

        result = ss.topk(arr, 9, True, False)

        np.testing.assert_array_equal(result, expected_arr)

    def test_raise_value_error_if_k_is_negative(self):
        arr = np.array([5, 4, 10, 23, 19, 1, 6, 15, 26])

        with self.assertRaises(ValueError):
            ss.topk(arr, -2, True, False)

    def test_raise_value_error_if_k_zero(self):
        
        arr = np.array([5, 4, 10, 23, 19, 1, 6, 15, 26])

        with self.assertRaises(ValueError):
            ss.topk(arr, 0, True, False)

    def test_raise_value_error_if_k_bigger_than_num_of_values(self):

        arr = np.array([5, 4, 10, 23, 19, 1, 6, 15, 26])

        with self.assertRaises(ValueError):
            ss.topk(arr, 10, True, False)

    def test_raise_value_error_if_arr_empty(self):
        arr = np.array([])

        with self.assertRaises(ValueError):
            ss.topk(arr, 0, True, False)

    def test_raise_value_error_if_nan_found(self):
        arr = np.array([5, 4, 10, np.nan, 19, 1, 6, np.nan, 26])

        with self.assertRaises(ValueError):
            ss.topk(arr, 7, True, False)


class TestQuickSelect(unittest.TestCase):
    def test_middle_k(self):
        arr = np.array([7, 2, 9, 4, 1, 6])

        result = ss.quickselect(arr, 3)

        np.testing.assert_array_equal(result, 4)

    def test_largest_value(self):
        arr = np.array([7, 2, 9, 4, 1, 6])

        result = ss.quickselect(arr, 6)

        np.testing.assert_array_equal(result, 9)

    def test_smallest_value(self):
        arr = np.array([7, 2, 9, 4, 1, 6])

        result = ss.quickselect(arr, 1)

        np.testing.assert_array_equal(result, 1)

    def test_duplicates(self):

        arr = np.array([7, 4, 9, 4, 1, 6])

        result = ss.quickselect(arr, 2)

        np.testing.assert_array_equal(result, 4)

    def test_all_equal(self):

        arr = np.array([4, 4, 4, 4, 4, 4])

        result = ss.quickselect(arr, 2)

        np.testing.assert_array_equal(result, 4)

    def test_sorted_array(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7])

        result = ss.quickselect(arr, 3)

        np.testing.assert_array_equal(result, 3)

    def test_reverse_sorted_array(self):
        arr = np.array([7, 6, 5, 4, 3, 2, 1])

        result = ss.quickselect(arr, 3)

        np.testing.assert_array_equal(result, 3)

    def test_raise_value_error_if_array_empty(self):
        arr = np.array([])
        with self.assertRaises(ValueError):
            ss.quickselect(arr, 1)

    def test_raise_value_error_if_k_negative(self):
        arr = np.array([[7, 6, 5, 4, 3, 2, 1]])
        with self.assertRaises(ValueError):
            ss.quickselect(arr, -1)

    def test_raise_value_error_if_k_too_large(self):
        arr = np.array([[7, 6, 5, 4, 3, 2, 1]])
        with self.assertRaises(ValueError):
            ss.quickselect(arr, 10)

    def test_raise_value_error_if_k_zero(self):
        arr = np.array([[7, 6, 5, 4, 3, 2, 1]])
        with self.assertRaises(ValueError):
            ss.quickselect(arr, 0)

    def test_raise_value_error_if_array_is_two_d(self):
        arr = np.array([
            [1, 2],
            [3, 4]
        ])

        with self.assertRaises(ValueError):
            ss.quickselect(arr, 2)
    

class TestBinarySearch(unittest.TestCase):
    def test_value_found_in_middle(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        result = ss.binary_search(arr, 5)

        self.assertEqual(result, (4, True))

    def test_value_found_at_start(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        result = ss.binary_search(arr, 1)

        self.assertEqual(result, (0, True))

    def test_value_found_at_end(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        result = ss.binary_search(arr, 9)

        self.assertEqual(result, (8, True))

    def test_value_not_found_at_end(self):
        arr = np.array([1, 3, 5, 7, 9])

        result = ss.binary_search(arr, 4)

        self.assertEqual(result, (2, False))
    
    def test_value_larger_then_rest(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        result = ss.binary_search(arr, 10)

        self.assertEqual(result, (9, False))

    def test_value_smaller_then_rest(self):
        arr = np.array([2, 3, 4, 5, 6, 7, 8, 9])

        result = ss.binary_search(arr, 1)

        self.assertEqual(result, (0, False))

    def test_single_element_array(self):
        arr = np.array([2])

        result = ss.binary_search(arr, 2)

        self.assertEqual(result, (0, True))

    def test_single_element_array_not_found(self):
        arr = np.array([2])

        result = ss.binary_search(arr, 3)

        self.assertEqual(result, (1, False))

    def test_empty_array(self):
        arr = np.array([])

        result = ss.binary_search(arr, 3)

        self.assertEqual(result, (0, False))
    
if __name__ == "__main__":
    unittest.main()
