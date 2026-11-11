"""
Unit tests for the tree.py module
"""

import unittest

import numpy as np

from numcompute.tree import DecisionTreeClassifier


class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[2.0], [4.0], [6.0], [8.0]])

        self.y = np.array([0, 0, 1, 1])

    def test_fit_predict(self):
        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(self.X, self.y)

        pred = tree.predict(self.X)

        np.testing.assert_array_equal(pred, self.y)

    def test_predict_before_fit(self):
        tree = DecisionTreeClassifier()

        with self.assertRaises(AttributeError):
            tree.predict(self.X)

    def test_partial_fit_before_buffer_reached(self):
        tree = DecisionTreeClassifier(buffer_size=4)

        tree.partial_fit(self.X[:2], self.y[:2])

        self.assertIsNone(tree.root)

    def test_partial_fit_retrains_when_buffer_reached(self):
        tree = DecisionTreeClassifier(buffer_size=4)

        tree.partial_fit(self.X[:2], self.y[:2])

        tree.partial_fit(self.X[2:], self.y[2:])

        self.assertIsNotNone(tree.root)

        pred = tree.predict(self.X)

        np.testing.assert_array_equal(pred, self.y)

    def test_buffer_is_cleared_after_model_retrained(self):
        tree = DecisionTreeClassifier(buffer_size=4)

        tree.partial_fit(self.X, self.y)

        self.assertIsNone(tree._X_buffer)
        self.assertIsNone(tree._y_buffer)

    def test_entropy_criterion(self):
        tree = DecisionTreeClassifier(max_depth=3, criterion="entropy")

        tree.fit(self.X, self.y)

        pred = tree.predict(self.X)

        np.testing.assert_array_equal(pred, self.y)

    def test_invalid_criterion(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(criterion="invalid")


if __name__ == "__main__":
    unittest.main()
