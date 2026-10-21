"""
    Unit tests for stats module
"""
import unittest

import numpy as np

from numcompute import mean, median, stan_dev, min, max, histogram, quantile

class statsTests(unittest.TestCase):
    def mean_test(self):
        test_data = np.array([10, 5, 20, 15])
        actual_mean = 12.5

        test_mean = mean(test_data)
        self.assertEqual(test_mean, actual_mean)
    def median_test(self):
        test_data = np.array([10 , 5, 20, 5, 7])
        actual_median = 7

        test_median = median(test_data)
        self.assertEqual(test_median, actual_median)
    def stan_dev_test(self):
        test_data = np.array([5, 10, 5 , 10, 5, 10])
        actual_sd = 2.5

        test_sd = stan_dev(test_data)
        self.assertEqual(test_sd, actual_sd)
    def min_test(self):
        test_data = np.array([5,10,20,30])
        actual_min = 5

        test_min = min(test_data)
        self.assertEqual(test_min, actual_min)
    def max_test(self):
        test_data = np.array([5,10,20,30])
        actual_max = 30

        test_max = max(test_data)
        self.assertEqual(test_max, actual_max)
    def hist_test(self):
        test_data = np.array([0,1,2,3])
        test_bins = 4
        actual_hist = (np.array[1,1,1,1],np.array[0,1,2,3])

        test_hist = histogram(test_data, test_bins)
        self.assertEqual(actual_hist, test_hist)
    def quan_test(self):
        test_data = np.array([0,1,2,3])
        test_prob = 0.5
        actual_div = 1.5

        test_div = quantile(test_data, test_prob)
        self.assertEqual(test_div, actual_div)
    