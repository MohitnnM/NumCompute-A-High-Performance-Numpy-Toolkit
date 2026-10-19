"""
    Unit tests for metrics.py module
"""


import unittest
import numpy as np
import sys
sys.path.append('../')
from numcompute.metrics import accuracy, precision, recall, f1score, confusion_matrix, MSE

class TestMetrics(unittest.TestCase):
    def test_correct_accuracy(self):
        pred_array = np.array([1,1,0,0])
        true_array = np.array([[1,0,0,0]])

        actual_accuracy = 0.75
        calc_accuracy = accuracy(pred_array, true_array)
        self.assertEqual(calc_accuracy, actual_accuracy)


