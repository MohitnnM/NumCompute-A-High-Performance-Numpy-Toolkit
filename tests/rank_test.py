"""
Unit tests for rank.py module
"""

import unittest

import numpy as np

from numcompute import percentile, rank


class TestRank(unittest.TestCase):
    def test_tied_values_by_method(self):
        arr = np.array([0, 2, 3, 2, 7, 2])

        np.testing.assert_array_equal(
            rank(arr, "average"), np.array([1.0, 3.0, 5.0, 3.0, 6.0, 3.0])
        )
        np.testing.assert_array_equal(
            rank(arr, "ordinal"), np.array([1.0, 2.0, 5.0, 3.0, 6.0, 4.0])
        )
        np.testing.assert_array_equal(
            rank(arr, "dense"), np.array([1.0, 2.0, 3.0, 2.0, 4.0, 2.0])
        )

    def test_single_data(self):
        arr = np.array([8])
        np.testing.assert_array_equal(rank(arr, "average"), np.array([1.0]))
        np.testing.assert_array_equal(rank(arr, "ordinal"), np.array([1.0]))
        np.testing.assert_array_equal(rank(arr, "dense"), np.array([1.0]))

    def test_reject_nodata(self):
        arr = np.array([])
        with self.assertRaises(ValueError):
            rank(arr)

    def test_reject_invalid_data(self):
        with self.assertRaises(TypeError):
            rank(["I'm not supposed to be here"])

    def test_sorted_data(self):
        sorted_arr = np.array([12, 24, 37, 45, 58, 61, 73, 89, 92, 100])

        np.testing.assert_array_equal(
            rank(sorted_arr, "average"),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        )
        np.testing.assert_array_equal(
            rank(sorted_arr, "ordinal"),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        )
        np.testing.assert_array_equal(
            rank(sorted_arr, "dense"),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        )


class TestPercentile(unittest.TestCase):
    def test_matches_numpy_percentile(self):
        arr = np.array([7, 1, 4, 4, 10, 2])

        for q in [0, 0.3, 0.5, 1]:
            for method in ["linear", "lower", "higher", "midpoint"]:
                with self.subTest(q=q, method=method):
                    expected = np.percentile(arr, q * 100, method=method)
                    self.assertAlmostEqual(percentile(arr, q, method), expected)

    def test_reject_nodata(self):
        arr = np.array([])
        q = 0.5
        with self.assertRaises(ValueError):
            percentile(arr, q)

    def test_reject_invalid_data(self):
        q = 0.5
        with self.assertRaises(TypeError):
            percentile(["I'm not supposed to be here"], q)

    def test_single_data(self):
        q = 0.5
        arr = [10]

        np.testing.assert_equal(percentile(arr, q), 10)

    def test_arr_with_equal_values(self):
        q = 0.5
        arr = [
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
        ]

        np.testing.assert_equal(percentile(arr, q), 10)

    def test_percentile_with_outlier(self):
        q = 0.5
        outlier_arr = [10, 12, 14, 15, 15, 16, 18, 20, 500]

        np.testing.assert_equal(percentile(outlier_arr, q), 15)


if __name__ == "__main__":
    unittest.main()
