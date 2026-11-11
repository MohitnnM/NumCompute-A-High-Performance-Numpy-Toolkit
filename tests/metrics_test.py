"""
Unit tests for metrics.py module
"""

import unittest

import numpy as np

from numcompute.metrics import (
    AUC,
    F1,
    MSE,
    Accuracy,
    ConfusionMatrix,
    Precision,
    Recall,
    RollingWindowMetric,
)


class TestAccuracy(unittest.TestCase):
    def test_correct_accuracy(self):
        pred_array = np.array([1, 1, 0, 0])
        true_array = np.array([1, 0, 0, 0])
        actual_accuracy = 0.75

        metric = Accuracy()
        metric.update(true_array, pred_array)

        self.assertEqual(metric.result(), actual_accuracy)

    def test_accuracy_across_chunks(self):
        metric = Accuracy()
        metric.update(np.array([1, 0]), np.array([1, 1]))
        metric.update(np.array([0, 0]), np.array([0, 0]))
        actual_accuracy = 0.75

        self.assertEqual(metric.result(), actual_accuracy)

    def test_accuracy_reset(self):
        metric = Accuracy()

        metric.update(np.array([1, 0]), np.array([1, 1]))

        metric.reset()

        self.assertEqual(metric.result(), 0.0)

    def test_accuracy_ignores_nan_pairs(self):
        pred_array = np.array([1, 1, np.nan, 0])
        true_array = np.array([1, np.nan, 0, 0])

        metric = Accuracy()
        metric.update(true_array, pred_array)

        self.assertEqual(metric.result(), 1.0)


class TestbinaryMetrics(unittest.TestCase):
    def setUp(self):
        self.pred_array = np.array([1, 1, 0, 0])
        self.true_array = np.array([1, 0, 0, 0])

    def test_correct_precision(self):
        metric = Precision()
        metric.update(self.true_array, self.pred_array)

        actual_precision = 0.5

        self.assertEqual(metric.result(), actual_precision)

    def test_correct_recall(self):
        metric = Recall()
        metric.update(self.true_array, self.pred_array)

        actual_recall = 1.0

        self.assertEqual(metric.result(), actual_recall)

    def test_correct_f1(self):
        metric = F1()
        metric.update(self.true_array, self.pred_array)

        actual_f1 = 2 / 3

        self.assertEqual(metric.result(), actual_f1)
    
    def test_f1_all_predictions_wrong(self):
        metric = F1()
        metric.update(np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]))

        self.assertEqual(metric.result(), 0.0)

    def test_correct_confusion_matrix(self):
        metric = ConfusionMatrix()
        metric.update(self.true_array, self.pred_array)

        actual_matrix = (1, 1, 2, 0)

        self.assertEqual(metric.result(), actual_matrix)

    def test_binary_metric_across_chunks(self):
        metric = Precision()

        metric.update(np.array([1, 0]), np.array([1, 1]))
        metric.update(np.array([1, 0]), np.array([1, 0]))

        # TP = 2, FP = 1
        self.assertAlmostEqual(metric.result(), 2 / 3)


class TestMSE(unittest.TestCase):
    def test_corect_mse(self):
        pred_array = np.array([2, 1, 0.5, 0])
        true_array = np.array([1, 0.5, 1, 0])
        actual_mse = 0.375

        metric = MSE()
        metric.update(true_array, pred_array)

        self.assertEqual(metric.result(), actual_mse)

    def test_nan_mse(self):
        pred_array = np.array([2, np.nan, 1, 1, 0.5, 0])
        true_array = np.array([1, 1, np.nan, 0.5, 1, 0])

        actual_mse = 0.375

        metric = MSE()
        metric.update(true_array, pred_array)

        self.assertEqual(metric.result(), actual_mse)

    def test_mse_across_chunks(self):
        metric = MSE()

        metric.update(np.array([1.0, 2.0]), np.array([1.0, 3.0]))

        metric.update(np.array([3.0, 4.0]), np.array([3.0, 5.0]))

        self.assertEqual(metric.result(), 0.5)


class TestAUC(unittest.TestCase):
    def test_perfect_auc(self):
        score_array = np.array([0.1, 0.2, 0.8, 0.9])
        true_array = np.array([0, 0, 1, 1])

        metric = AUC()
        metric.update(true_array, score_array)

        self.assertEqual(metric.result(), 1.0)

    def test_auc_across_chunks(self):
        metric = AUC()
        metric.update(np.array([0, 0]), np.array([0.1, 0.2]))

        metric.update(np.array([1, 1]), np.array([0.8, 0.9]))

        self.assertEqual(metric.result(), 1.0)

    def test_auc_single_class_returns_zero(self):
        metric = AUC()
        metric.update(np.array([1, 1, 1]), np.array([0.4, 0.3, 0.6]))

        self.assertEqual(metric.result(), 0.0)


class TestRollingWindowMetric(unittest.TestCase):
    def test_rolling_accuracy(self):
        metric = RollingWindowMetric(Accuracy(), window_size=3)

        metric.update(np.array([1, 0]), np.array([1, 1]))

        metric.update(np.array([1, 0]), np.array([0, 0]))

        self.assertAlmostEqual(metric.result(), 1 / 3)

    def test_rolling_metric_reset(self):
        metric = RollingWindowMetric(Accuracy(), window_size=3)

        metric.update(np.array([1, 0]), np.array([1, 1]))

        metric.reset()

        self.assertAlmostEqual(metric.result(), 0.0)


if __name__ == "__main__":
    unittest.main()
