"""
    Unit tests for the io.py module
"""

import unittest

import numpy as np

from numcompute.io import load_csv


class TestLoadCSV(unittest.TestCase):
    def test_load_csv(self):
        file = "tests/test_csv_files/general_file.csv"

        expected_array = np.array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
            [13,14,15],
            [16,17,18],
            [19,20,21]
        ])

        resulting_array = load_csv(file)

        np.testing.assert_array_equal(resulting_array, expected_array)

    def test_array(self):
        file = "tests/test_csv_files/general_file.csv"


        resulting_array = load_csv(file)

        self.assertIsInstance(resulting_array, np.ndarray)

    def test_custom_delimiter_tab(self):
        file = "tests/test_csv_files/tab_delimiter_test.csv"

        expected_array = np.array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
            [13,14,15],
            [16,17,18],
            [19,20,21]
        ])

        resulting_array = load_csv(file, file_delimiter="\t")

        np.testing.assert_array_equal(resulting_array, expected_array)

    def test_custom_delimiter_semicolon(self):
        file = "tests/test_csv_files/semicolon_delimiter_test.csv"

        expected_array = np.array([
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
            [13,14,15],
            [16,17,18],
            [19,20,21]
        ])

        resulting_array = load_csv(file, ";")

        np.testing.assert_array_equal(resulting_array, expected_array)

    def test_single_missing_value(self):
        file = "tests/test_csv_files/missing_data_single.csv"

        expected_array = np.array([
            [1,2,np.nan],
            [4,5,6],
            [7,8,9],
            [10,11,12],
            [13,14,15],
            [16,17,18],
            [19,20,21]
        ])

        resulting_array = load_csv(file)

        np.testing.assert_array_equal(resulting_array, expected_array)

    def test_missing_values_test(self):
        file = "tests/test_csv_files/missing_data_multiple.csv"

        expected_array = np.array([
            [1,2,np.nan],
            [4,5,6],
            [7,np.nan,9],
            [10,11,12],
            [13,14,15],
            [16,17,np.nan],
            [19,20,21]
        ])

        resulting_array = load_csv(file)

        np.testing.assert_array_equal(resulting_array, expected_array)

    def test_negative_float_numbers(self):
        file = "tests/test_csv_files/negative_floating_numbers_test.csv"

        expected_array = np.array([
            [1,2,-3],
            [4,-5,6],
            [7,8,9.4],
            [-10.23,11,12],
            [13,14.6,15],
            [16,-17,18],
            [19,20.1,21]
        ])

        resulting_array = load_csv(file)

        np.testing.assert_array_equal(resulting_array, expected_array)

    def test_invalid_path(self):

        non_existing_file = "a_test_path.csv"
        with self.assertRaises(FileNotFoundError):
            load_csv(non_existing_file)

    def test_empty_file(self):
        file = "tests/test_csv_files/empty_file.csv"

        with self.assertRaises(ValueError):
            load_csv(file)

    def test_shape(self):
        file = "tests/test_csv_files/test_shape_file.csv"

        result = load_csv(file)

        np.testing.assert_array_equal(result.shape,[1,3])
if __name__ == "__main__":
    unittest.main()
