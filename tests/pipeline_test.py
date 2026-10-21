"""
Unit tests for the pipeline.py module
"""
import unittest
import numpy as np

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler, MinMaxScaler

class DummyTransformerNoFit:
    def transform(self, X):
        return X

class DummyTransformerNoTransform:
    def fit(self, X):
        return self


class DummyEstimator:
    def fit(self, X):
        self.fitted = True
        return self

    def predict(self, X):
        return X[:, 0]

class TestPipeline(unittest.TestCase):

    def test_transformer_pipeline_is_valid(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("minmax", MinMaxScaler())
        ])

        self.assertEqual(len(pipe.steps), 2)
        self.assertFalse(pipe.fitted)

    def test_fit_transform_with_preprocessing(self):
        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0]
        ])

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("minmax", MinMaxScaler())
        ])

        expected_arr = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0]
        ])

        result = pipe.fit_transform(X)

        np.testing.assert_allclose(result, expected_arr)
        self.assertTrue(pipe.fitted)

    def test_raises_value_error_if_empty_pipeline_steps(self):

        with self.assertRaises(ValueError):
            Pipeline([])

    def test_raises_value_error_if_steps_not_a_pair(self):
        with self.assertRaises(ValueError):
            Pipeline([
                ("scale", StandardScaler(), "extra step")
            ])

    def test_raises_type_error_if_step_not_tuple_or_list(self):
         with self.assertRaises(TypeError):
            Pipeline([
                StandardScaler()
            ])

    def test_raises_type_error_if_step_name_is_not_string(self):
        with self.assertRaises(TypeError):
            Pipeline([
                (23, StandardScaler())
            ])

    def test_raises_value_error_if_duplicate_step_names_present(self):
        with self.assertRaises(ValueError):
            Pipeline([
                ("scale", StandardScaler()),
                ("scale", StandardScaler())
            ])
    
    def test_raises_attribute_error_if_step_has_no_fit_method(self):
        with self.assertRaises(AttributeError):
            Pipeline([
                ("bad", DummyTransformerNoFit()),
                ("scale", StandardScaler())
            ])

    def test_raises_attribute_error_if_last_step_has_no_fit_method(self):
        with self.assertRaises(AttributeError):
            Pipeline([
                ("scale", StandardScaler()),
                ("bad", DummyTransformerNoFit())
            ])

    def test_raises_attribute_error_if_last_step_has_no_transform_method(self):
        with self.assertRaises(AttributeError):
            Pipeline([
                ("scale", StandardScaler()),
                ("bad", DummyTransformerNoTransform())
            ])
    
    def test_raises_attribute_error_if_step_has_no_transform_method(self):
        with self.assertRaises(AttributeError):
            Pipeline([
                ("bad", DummyTransformerNoTransform()),
                ("scale", StandardScaler())
            ])


    def test_raises_runtime_error_if_attempt_to_transform_before_fit(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0]
        ])

        pipe = Pipeline([
                ("scale", StandardScaler())
        ])

        with self.assertRaises(RuntimeError):
            pipe.transform(X)

    
    def test_if_fit_method_sets_fitted_to_true_and_return_self(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0] 
        ])

        pipe = Pipeline([
                ("scale", StandardScaler())
        ])

        result = pipe.fit(X)

        self.assertTrue(pipe.fitted)
        self.assertIs(result, pipe)

    def test_single_step_pipeline(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0] 
        ])

        pipe = Pipeline([
                ("scale", StandardScaler())
        ])

        result = pipe.fit_transform(X)
        expected_arr = StandardScaler().fit_transform(X)

        np.testing.assert_allclose(result, expected_arr)


    def test_raises_runtime_error_if_attempt_to_predict_before_fit(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0]
        ])

        pipe = Pipeline([
                ("scale", StandardScaler())
        ])

        with self.assertRaises(RuntimeError):
            pipe.predict(X)

    
    def test_raises_attribute_error_if_predict_called_on_transformer(self):
        X = np.array([
            [1.0],
            [2.0],
            [3.0]
        ])

        pipe = Pipeline([
                ("scale", StandardScaler())
        ])

        pipe.fit(X)

        with self.assertRaises(AttributeError):
            pipe.predict(X)

    
    def test_if_prediction_correct_for_estimator_pipeline(self):
        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0]
        ])

        scaler = StandardScaler()

        pipe = Pipeline([
            ("scale", scaler),
            ("model", DummyEstimator())
        ])

        pipe.fit(X)
        result = pipe.predict(X)

        transformed_X = scaler.transform(X)
        expected = transformed_X[:, 0]

        np.testing.assert_allclose(result, expected)
        

    def test_if_transform_does_not_refit(self):
        X_train = np.array([
            [1.0],
            [2.0],
            [3.0]
        ])

        X_test = np.array([
            [10.0],
            [20.0]
        ])

        scaler = StandardScaler()

        pipe = Pipeline([
            ("scale", scaler)
        ])

        pipe.fit(X_train)
        initial_mean = scaler.mean_.copy()
        initial_scale = scaler.scale_.copy()

        pipe.transform(X_test)

        np.testing.assert_array_equal(scaler.mean_, initial_mean)

        np.testing.assert_array_equal(scaler.scale_, initial_scale)