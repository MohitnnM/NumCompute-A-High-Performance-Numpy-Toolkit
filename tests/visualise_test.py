"""
Unit test for the visualise.py module
"""

import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from numcompute.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


class TestVisualise(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_metric_over_time_returns_figure_and_axis(self):
        metric_values = [0.5, 0.6, 0.7, 0.8]

        figure, axis = plot_metric_over_time(
            metric_values, "Accuracy Over Time", "Accuracy", show=False
        )

        self.assertIsNotNone(figure)
        self.assertIsNotNone(axis)

        self.assertEqual(axis.get_title(), "Accuracy Over Time")
        self.assertEqual(axis.get_ylabel(), "Accuracy")

    def test_compare_models_returns_figure_and_axis(self):
        metric1 = [0.5, 0.6, 0.7]
        metric2 = [0.4, 0.55, 0.8]

        figure, axis = compare_models(
            metric1, metric2, ["Model 1", "Model 2"], show=False
        )

        self.assertIsNotNone(figure)
        self.assertIsNotNone(axis)

        self.assertEqual(len(axis.lines), 2)

    def test_compare_models_different_len_error(self):
        metric1 = [0.5, 0.6, 0.7]
        metric2 = [0.4, 0.55]

        with self.assertRaises(ValueError):
            compare_models(metric1, metric2, ["Model 1", "Model 2"], show=False)

    def test_compare_models_invalid_labels(self):
        metric1 = [0.5, 0.6]
        metric2 = [0.4, 0.55]

        with self.assertRaises(ValueError):
            compare_models(metric1, metric2, ["Model 1"], show=False)

    def test_plot_predictions_vs_ground_truth_returns_figure_and_axis(self):
        true_array = np.array([0, 1, 1, 0])
        pred_array = np.array([0, 1, 0, 0])

        figure, axis = plot_predictions_vs_ground_truth(
            true_array, pred_array, show=False
        )

        self.assertIsNotNone(figure)
        self.assertIsNotNone(axis)

        self.assertEqual(len(axis.lines), 2)

    def test_plot_predictions_with_different_shapes_error(self):
        true_array = np.array([0, 1, 1, 0])
        pred_array = np.array([0, 1, 1])

        with self.assertRaises(ValueError):
            plot_predictions_vs_ground_truth(true_array, pred_array, show=False)

    def test_plots_can_be_saved(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_dir:
            path = temp_dir.name

        os.remove(path)

        plot_metric_over_time(
            [0.5, 0.6, 0.7], "Accuracy", "Accuracy", save_path=path, show=False
        )

        self.assertTrue(os.path.exists(path))

        os.remove(path)


if __name__ == "__main__":
    unittest.main()
