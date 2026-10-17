"""
Unit tests for the optim.py module
"""

import unittest

import numpy as np

from numcompute.optim import grad, jacobian


def sum_of_squares(x):
    return np.sum(x**2)


class TestGrad(unittest.TestCase):
    def test_central_matches_analytic(self):
        x = np.array([1.0, 2.0, 3.0])

        result = grad(sum_of_squares, x)

        np.testing.assert_allclose(result, 2 * x, rtol=1e-8)

    def test_forward_matches_analytic(self):
        x = np.array([1.0, 2.0, 3.0])

        result = grad(sum_of_squares, x, method="forward")

        # Forward difference is less accurate (error O(h)).
        np.testing.assert_allclose(result, 2 * x, rtol=1e-4)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            grad(sum_of_squares, [1.0, 2.0], method="backward")

    def test_vector_output_raises(self):
        with self.assertRaises(ValueError):
            grad(lambda x: x * 2, [1.0, 2.0])


class TestJacobian(unittest.TestCase):
    def test_matches_analytic(self):
        def vector_func(x):
            return np.array([x[0] * x[1], np.sin(x[0])])

        x = np.array([1.0, 2.0])
        expected = np.array([[x[1], x[0]], [np.cos(x[0]), 0.0]])

        result = jacobian(vector_func, x)

        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
