"""
    Unit tests for rank.py module
"""

import unittest

import numpy as np

import numcompute.rank as rk


class TestRank(unittest.TestCase):

    def test_tied_values_by_method(self):
        arr = np.array([0, 2, 3, 2, 7, 2])

        np.testing.assert_array_equal(
            rk.rank(arr, "average"), np.array([1, 3, 5, 3, 6, 3])
        )
        np.testing.assert_array_equal(
            rk.rank(arr, "ordinal"), np.array([1, 2, 5, 3, 6, 4])
        )
        np.testing.assert_array_equal(
            rk.rank(arr, "dense"), np.array([1, 2, 3, 2, 4, 2])
        )


class TestPercentile(unittest.TestCase):

    def test_matches_numpy_percentile(self):
        arr = np.array([7, 1, 4, 4, 10, 2])

        for q in [0, 0.3, 0.5, 1]:
            for method in ["linear", "lower", "higher", "midpoint"]:
                with self.subTest(q=q, method=method):
                    expected = np.percentile(arr, q * 100, method=method)
                    self.assertAlmostEqual(rk.percentile(arr, q, method), expected)


if __name__ == "__main__":
    unittest.main()
