"""
Unit tests for the benchmarking.py module.
"""

import unittest

import numpy as np

from numcompute.benchmarking import benchmark_buffer_sizes, benchmark_streaming_scaler


class TestBenchMarking(unittest.TestCase):
    def test_streaming_scaler_benchmark_return_results(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

        results = benchmark_streaming_scaler(X, size_of_chunk=2, repeat=1)

        self.assertIn("loop_times_ms", results)
        self.assertIn("vectorised_times_ms", results)
        self.assertIn("loop_average_ms", results)
        self.assertIn("vectorised_average_ms", results)
        self.assertIn("speed_up", results)

    def test_streaming_scaler_benchamrk_times_not_negative(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        results = benchmark_streaming_scaler(X, size_of_chunk=2, repeat=1)

        self.assertGreaterEqual(results["loop_average_ms"], 0)
        self.assertGreaterEqual(results["vectorised_average_ms"], 0)

    def test_buffer_size_benchmark_returns_each_buffer_size(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

        y = np.array([0, 0, 0, 1, 1, 1])

        buffer_size = [2, 3, 6]

        results = benchmark_buffer_sizes(
            X, y, buffer_sizes=buffer_size, size_of_chunk=2, repeat=2, max_depth=2
        )

        self.assertEqual(len(results), len(buffer_size))

    def test_buffer_size_benchmark_contains_expected_values(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])

        y = np.array([0, 0, 0, 1, 1, 1])

        results = benchmark_buffer_sizes(
            X, y, buffer_sizes=[2, 6], size_of_chunk=2, repeat=2, max_depth=2
        )

        for result in results:
            self.assertIn("buffer_size", result)
            self.assertIn("avg_time_ms", result)
            self.assertIn("accuracy", result)


if __name__ == "__main__":
    unittest.main()
