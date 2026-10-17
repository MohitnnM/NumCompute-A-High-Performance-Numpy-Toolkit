"""
    Unit tests for the utils.py module
"""

import unittest

import numpy as np

from numcompute.exceptions import NotFittedError
from numcompute.utils import check_array, check_is_fitted


class TestCheckArray(unittest.TestCase):
    def test_valid_list_returns_float_array(self):
        result = check_array([[1, 2], [3, 4]])

        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.shape, (2, 2))

    def test_strings_rejected_by_default(self):
        with self.assertRaises(TypeError):
            check_array([["a", "b"], ["c", "d"]])

    def test_dtype_none_keeps_strings(self):
        result = check_array([["red"], ["blue"]], dtype=None)

        self.assertTrue(np.issubdtype(result.dtype, np.str_))

    def test_ragged_rows_rejected(self):
        with self.assertRaises(ValueError):
            check_array([[1, 2], [3]])

    def test_1d_rejected(self):
        with self.assertRaises(ValueError):
            check_array([1, 2, 3])

    def test_nan_rejected_by_default(self):
        with self.assertRaises(ValueError):
            check_array([[1, np.nan]])

    def test_nan_allowed_when_requested(self):
        result = check_array([[1, np.nan]], allow_nan=True)

        self.assertTrue(np.isnan(result[0, 1]))


class TestCheckIsFitted(unittest.TestCase):
    class Dummy:
        pass

    def test_unfitted_raises(self):
        with self.assertRaises(NotFittedError):
            check_is_fitted(self.Dummy(), "mean_")

    def test_fitted_passes(self):
        est = self.Dummy()
        est.mean_ = 0.0

        self.assertIsNone(check_is_fitted(est, "mean_"))


if __name__ == "__main__":
    unittest.main()
