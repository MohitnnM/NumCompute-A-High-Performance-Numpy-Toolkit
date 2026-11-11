"""
Unit tests for the preprocessing.py module
"""

import unittest

import numpy as np

from numcompute.exceptions import NotFittedError
from numcompute.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


class TestStandardScaler(unittest.TestCase):
    def test_output_has_zero_mean_unit_std(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        result = StandardScaler().fit_transform(X)

        np.testing.assert_allclose(result.mean(axis=0), [0, 0], atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), [1, 1])

    def test_constant_column_becomes_zeros(self):
        X = np.full((3, 1), 0.1)

        result = StandardScaler().fit_transform(X)

        np.testing.assert_allclose(result, np.zeros((3, 1)), atol=1e-12)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            StandardScaler().transform([[1.0, 2.0]])

    def test_wrong_feature_count_raises(self):
        scaler = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaises(ValueError):
            scaler.transform([[1.0, 2.0, 3.0]])

    def test_partial_fit_is_same_as_fit(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

        batch_scaler = StandardScaler().fit(X)

        stream_scaler = StandardScaler()
        stream_scaler.partial_fit(X[:2])
        stream_scaler.partial_fit(X[2:])

        np.testing.assert_allclose(stream_scaler.mean_, batch_scaler.mean_)

        np.testing.assert_allclose(stream_scaler.scale_, batch_scaler.scale_)

    def test_partial_fit__transform_is_same_as_fit(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

        batch_scaler = StandardScaler().fit(X)

        stream_scaler = StandardScaler()
        stream_scaler.partial_fit(X[:2])
        stream_scaler.partial_fit(X[2:])

        np.testing.assert_allclose(
            stream_scaler.transform(X), batch_scaler.transform(X)
        )
    
    def test_standard_scaler_tiny_varying_feature(self):
        X = np.array([[1e-10], [2e-10], [3e-10]])

        scaler = StandardScaler()
        result = scaler.fit_transform(X)

        self.assertNotEqual(scaler.scale_[0], 1.0)

        np.testing.assert_allclose(result.mean(axis=0), [0.0], atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0), [1.0], atol=1e-12)


class TestMinMaxScaler(unittest.TestCase):
    def test_default_range(self):
        X = np.array([[1.0], [2.0], [3.0]])

        result = MinMaxScaler().fit_transform(X)

        np.testing.assert_allclose(result, [[0.0], [0.5], [1.0]])

    def test_custom_range(self):
        X = np.array([[1.0], [2.0], [3.0]])

        result = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

        np.testing.assert_allclose(result, [[-1.0], [0.0], [1.0]])

    def test_constant_column_maps_to_range_min(self):
        X = np.full((3, 1), 5.0)

        result = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

        np.testing.assert_allclose(result, np.full((3, 1), -1.0))

    def test_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            MinMaxScaler(feature_range=(1, 0))

    def test_partial_fit_updates_min_and_max(self):
        scaler = MinMaxScaler()

        scaler.partial_fit(np.array([[1.0], [3.0]]))

        scaler.partial_fit(np.array([[2.0], [4.0]]))

        result = scaler.transform(np.array([[1.0], [3.0]]))

        np.testing.assert_allclose(result, [[0.0], [2.0 / 3.0]])


class TestOneHotEncoder(unittest.TestCase):
    def setUp(self):
        self.X = [["red", "S"], ["blue", "L"], ["red", "L"]]

    def test_encoding(self):
        # Categories are sorted: ["blue", "red"] and ["L", "S"].
        expected = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 1, 0],
            ]
        )

        result = OneHotEncoder().fit_transform(self.X)

        np.testing.assert_array_equal(result, expected)

    def test_unknown_category_raises(self):
        encoder = OneHotEncoder().fit(self.X)

        with self.assertRaises(ValueError):
            encoder.transform([["green", "S"]])

    def test_unknown_category_ignored(self):
        encoder = OneHotEncoder(handle_unknown="ignore").fit(self.X)

        result = encoder.transform([["green", "S"]])

        np.testing.assert_array_equal(result, [[0, 0, 0, 1]])

    def test_partial_fit_expands_categories(self):
        encoder = OneHotEncoder()

        encoder.partial_fit([["red"], ["blue"]])

        encoder.partial_fit([["green"]])

        result = encoder.transform([["red"], ["blue"], ["green"]])

        self.assertEqual(result.shape[1], 3)


class TestSimpleImputer(unittest.TestCase):
    def test_nan_replaced_with_feature_mean(self):
        X = np.array([[1.0, np.nan], [np.nan, 4.0]])

        result = SimpleImputer(fill_value=-1).fit_transform(X)

        np.testing.assert_array_equal(result, [[1.0, 4.0], [1.0, 4.0]])

    def test_input_not_modified(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])

        SimpleImputer().fit_transform(X)

        self.assertTrue(np.isnan(X[0, 1]))

    def test_non_numeric_fill_value_raises(self):
        with self.assertRaises(TypeError):
            SimpleImputer(fill_value="missing")

    def test_partial_fit_updates_mean(self):
        imputer = SimpleImputer()

        imputer.partial_fit(np.array([[1.0], [3.0]]))

        imputer.partial_fit(np.array([[5.0], [7.0]]))

        result = imputer.transform(np.array([[np.nan]]))

        np.testing.assert_allclose(result, [[4.0]])

    def test_partial_fit_ignored_nan_when_updating_mean(self):
        imputer = SimpleImputer()

        imputer.partial_fit(np.array([[2.0], [np.nan], [3.0]]))

        result = imputer.transform(np.array([[np.nan]]))

        np.testing.assert_allclose(result, [[2.5]])


if __name__ == "__main__":
    unittest.main()
