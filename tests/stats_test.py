"""
Unit tests for stats module
"""

import unittest

import numpy as np

from numcompute import (
    StreamingStats,
    histogram,
    max,
    mean,
    median,
    min,
    quantile,
    stan_dev,
)


class TestStats(unittest.TestCase):
    def test_mean(self):
        test_data = np.array([10, 5, 20, 15])
        actual_mean = 12.5

        test_mean = mean(test_data)
        self.assertEqual(test_mean, actual_mean)

    def test_median(self):
        test_data = np.array([10, 5, 20, 5, 7])
        actual_median = 7

        test_median = median(test_data)
        self.assertEqual(test_median, actual_median)

    def test_stan_dev(self):
        test_data = np.array([5, 10, 5, 10, 5, 10])
        actual_sd = 2.5

        test_sd = stan_dev(test_data)
        self.assertEqual(test_sd, actual_sd)

    def test_min(self):
        test_data = np.array([5, 10, 20, 30])
        actual_min = 5

        test_min = min(test_data)
        self.assertEqual(test_min, actual_min)

    def test_max(self):
        test_data = np.array([5, 10, 20, 30])
        actual_max = 30

        test_max = max(test_data)
        self.assertEqual(test_max, actual_max)

    def test_hist(self):
        test_data = np.array([0, 1, 2, 3])
        test_bins = 4
        actual_hist = (np.array([1, 1, 1, 1]), np.array([0, 0.75, 1.5, 2.25, 3]))

        test_hist = histogram(test_data, test_bins)
        np.testing.assert_array_equal(actual_hist[0], test_hist[0])
        np.testing.assert_array_equal(actual_hist[1], test_hist[1])

    def test_quan(self):
        test_data = np.array([0, 1, 2, 3])
        test_prob = 0.5
        actual_div = 1.5

        test_div = quantile(test_data, test_prob)
        self.assertEqual(test_div, actual_div)

    def test_nan_max(self):
        test_data = np.array([5, 10, 20, np.nan, 30])
        actual_max = 30

        test_max = max(test_data)
        self.assertEqual(test_max, actual_max)

    def test_nan_quan(self):
        test_data = np.array([0, 1, 2, 3, np.nan])
        test_prob = 0.5
        actual_div = 1.5

        test_div = quantile(test_data, test_prob)
        self.assertEqual(test_div, actual_div)

    def test_nan_hist(self):
        test_data = np.array([0, 1, 2, 3, np.nan])
        test_bins = 4
        actual_hist = (np.array([1, 1, 1, 1]), np.array([0, 0.75, 1.5, 2.25, 3]))

        test_hist = histogram(test_data, test_bins)
        np.testing.assert_array_equal(actual_hist[0], test_hist[0])
        np.testing.assert_array_equal(actual_hist[1], test_hist[1])


class TestStreamingStats(unittest.TestCase):
    def test_streaming_mean_across_chunks(self):
        stat = StreamingStats()

        stat.update_stats(np.array([[1.0], [2.0]]))

        stat.update_stats(np.array([[3.0], [4.0]]))

        np.testing.assert_allclose(stat.mean(), [2.5])

    def test_streaming_variance_across_chunks(self):
        stat = StreamingStats()

        stat.update_stats(np.array([[1.0], [2.0]]))

        stat.update_stats(np.array([[3.0], [4.0]]))

        np.testing.assert_allclose(stat.variance(), [1.25])

    def test_streaming_stan_dev_across_chunks(self):
        stat = StreamingStats()

        stat.update_stats(np.array([[1.0], [2.0], [3.0], [4.0]]))

        np.testing.assert_allclose(stat.stan_dev(), np.sqrt(1.25))

    def test_streaming_min_max_across_chunks(self):
        stat = StreamingStats()

        stat.update_stats(np.array([[15.0, 17.0], [18.0, 25.0]]))

        stat.update_stats(np.array([[3.0, 27.0], [8, 16.0]]))

        np.testing.assert_array_equal(stat.min(), [3.0, 16.0])
        np.testing.assert_array_equal(stat.max(), [18.0, 27.0])

    def test_streaming_ignores_nan(self):
        stat = StreamingStats()

        stat.update_stats(np.array([[1.0], [np.nan]]))

        stat.update_stats(np.array([[3.0]]))

        np.testing.assert_allclose(stat.mean(), [2.0])

    def test_streaming_meadian_across_chunks(self):
        stat = StreamingStats()

        stat.update_stats(np.array([[1.0], [4.0]]))

        stat.update_stats(np.array([[2.0], [3.0]]))

        np.testing.assert_allclose(stat.median(), [2.5])

    def test_streaming_quantile_across_chunks(self):
        stat = StreamingStats()

        stat.update_stats(np.array([[1.0], [2.0]]))

        stat.update_stats(np.array([[3.0], [4.0]]))

        np.testing.assert_allclose(stat.quantile(0.5), [2.5])

    def test_streaming_histogram(self):
        stat = StreamingStats()

        stat.update_stats(np.array([[0.0], [1.0]]))

        stat.update_stats(np.array([[2.0], [3.0]]))

        count, edges = stat.histogram(bins=4)

        np.testing.assert_array_equal(count, [1, 1, 1, 1])

    def test_reset(self):
        stat = StreamingStats()

        stat.update_stats(np.array([[1.0], [4.0]]))

        stat.reset()

        self.assertIsNone(stat.mean())


if __name__ == "__main__":
    unittest.main()
