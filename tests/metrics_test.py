"""
    Unit tests for metrics.py module
"""
import unittest
import numpy as np
from numcompute.metrics import accuracy, precision, recall, f1score, confusion_matrix, MSE

class TestMetrics(unittest.TestCase):
    def test_correct_accuracy(self):
        pred_array = np.array([1,1,0,0])
        true_array = np.array([1,0,0,0])
        actual_accuracy = 0.75

        calc_accuracy = accuracy(pred_array, true_array)

        self.assertEqual(calc_accuracy, actual_accuracy)
    def test_correct_precision(self):
        pred_array = np.array([1,1,0,0])
        true_array = np.array([1,0,0,0])
        actual_precision = 0.5

        calc_precision = precision(pred_array, true_array)

        self.assertEqual(calc_precision,actual_precision)
    def test_correct_recall(self):
        pred_array = np.array([1,1,0,0])
        true_array = np.array([1,0,0,0])
        actual_recall = 1

        calc_recall = recall(pred_array, true_array)

        self.assertEqual(calc_recall,actual_recall)
    def test_correct_f1score(self):
        pred_array = np.array([1,1,0,0])
        true_array = np.array([1,0,0,0])
        actual_f1score = 2/3

        calc_f1score = f1score(pred_array, true_array)

        self.assertEqual(calc_f1score,actual_f1score)
    def test_correct_confusion_matrix(self):
        pred_array = np.array([1,1,0,0])
        true_array = np.array([1,0,0,0])
        actual_matrix = (1,1,2,0)

        calc_matrix = confusion_matrix(pred_array, true_array)

        self.assertEqual(calc_matrix, actual_matrix)
    def test_corect_MSE(self):
        pred_array = np.array([2,1,0.5,0])
        true_array = np.array([1,0.5,1,0])
        actual_mse = 0.375

        calc_mse = MSE(pred_array,true_array)

        self.assertEqual(calc_mse,actual_mse)
    def test_nan_MSE(self):
        pred_array = np.array([2,np.nan,1,1,0.5,0])
        true_array = np.array([1,1,np.nan,0.5,1,0])
        actual_mse = 0.375

        calc_mse = MSE(pred_array,true_array)

        self.assertEqual(calc_mse,actual_mse)
    def test_nan_prec(self):
        pred_array = np.array([1,np.nan,1,1,0,0])
        true_array = np.array([1,1,np.nan,0,0,0])
        actual_precision = 0.5

        calc_precision = precision(pred_array, true_array)

        self.assertEqual(calc_precision,actual_precision)
    def test_nan_con_mat(self):
        pred_array = np.array([1,np.nan,1,1,0,0])
        true_array = np.array([1,1,np.nan,0,0,0])
        actual_matrix = (1,1,2,0)

        calc_matrix = confusion_matrix(pred_array, true_array)

        self.assertEqual(calc_matrix, actual_matrix)
    def test_nan_f1(self):
        pred_array = np.array([1,np.nan,1,0,0])
        true_array = np.array([1,1,0,0,0])
        actual_f1score = 2/3

        calc_f1score = f1score(pred_array, true_array)

        self.assertEqual(calc_f1score,actual_f1score)
if __name__ == "__main__":
    unittest.main()


