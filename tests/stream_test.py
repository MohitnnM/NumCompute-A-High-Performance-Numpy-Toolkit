"""
Unit test for the stream.py module
"""

import unittest

import numpy as np

from numcompute.metrics import Accuracy
from numcompute.stream import StreamTrainer


class ModelDummy:
    def __init__(self):
        self.fitted = False

    def partial_fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        return np.asarray([1, 0, 1, 0])[: X.shape[0]]


class TestStreamTrainer(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0], [2.0], [3.0], [4.0]])

        self.y = np.array([1, 0, 1, 1])

    def test_accuracy_is_default_metric(self):
        trainer = StreamTrainer(ModelDummy())

        self.assertIsInstance(trainer.metric, Accuracy)

    def test_fit_chunk_updates_model_and_chunk_count(self):
        model = ModelDummy()
        trainer = StreamTrainer(model)

        result = trainer.fit_chunk(self.X, self.y)

        self.assertTrue(model.fitted)
        self.assertEqual(trainer.seen_chunks_, 1)
        self.assertIs(result, trainer)

    def test_score_chunk_returns_log(self):

        trainer = StreamTrainer(ModelDummy())

        trainer.fit_chunk(self.X, self.y)

        log = trainer.score_chunk(self.X, self.y)

        self.assertIn("chunk", log)
        self.assertIn("chunk_acc", log)
        self.assertIn("cum_acc", log)
        self.assertIn("memory_usage", log)

    def test_score_chunk_accuracy(self):
        trainer = StreamTrainer(ModelDummy())

        trainer.fit_chunk(self.X, self.y)

        log = trainer.score_chunk(self.X, self.y)

        self.assertEqual(log["chunk_acc"], 0.75)
        self.assertEqual(log["cum_acc"], 0.75)

    def test_log_accumulate(self):
        trainer = StreamTrainer(ModelDummy())

        trainer.fit_chunk(self.X, self.y)
        trainer.score_chunk(self.X, self.y)

        trainer.fit_chunk(self.X, self.y)
        trainer.score_chunk(self.X, self.y)

        self.assertEqual(len(trainer.get_logs()), 2)

    def test_reset(self):
        trainer = StreamTrainer(ModelDummy())

        trainer.fit_chunk(self.X, self.y)
        trainer.score_chunk(self.X, self.y)

        trainer.reset()

        self.assertEqual(trainer.seen_chunks_, 0)

        self.assertEqual(trainer.logs, [])

        self.assertEqual(trainer.metric.result(), 0.0)


if __name__ == "__main__":
    unittest.main()
