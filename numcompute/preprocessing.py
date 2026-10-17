"""
Transforms pre-loaded arrays into data suitable for analysis.

This module provides transformations that prepare both numerical and
categorical data for further analysis or modelling. Components included
are feature standardisation, scaling, and categorical encoding.

Transformers follow a fit-then-transform style. During fit(X), each
transformer learns and stores the parameters or categories required for
new data. transform(X) applies the learned state to new data without
relearning it. NumPy vectorisation and broadcasting are used to implement
all numerical operations.

Classes Included:
  - `StandardScaler` (z-score standardisation)
  - `MinMaxScaler` (scale to a given range)
  - `OneHotEncoder` (for categorical variables)
  - `SimpleImputer` (replace NaN with a constant)

Implementation requirements:
- Core transformations should be implemented using vectorised NumPy operations,
  not Python loops where possible.
"""

import numpy as np

from numcompute.utils import check_array, check_is_fitted


class StandardScaler:
    """
    Standardises numerical features, centres them by subtracting their means
    and scales them by dividing the difference by their standard deviation.

    APIs implemented:
      - `fit(X) -> self`: calculates and stores the mean and standard
         deviation per feature.

      - `transform(X) -> X_out`: Uses the stored values from fit(X) to
         standardise data.

      - `fit_transform(X) -> X_out`: fits the transformer to X, then
         transforms using the learned state.
    """

    def fit(self, X):
        """
        Computes the per-feature mean and standard deviation.

        Constant features get a scale of 1, so they become all zeros
        after transform instead of dividing by zero.

        Parameters:
            X:
                2D array-like of shape (n_samples, n_features).

        Returns:
            self, with mean_, scale_ and n_features_in_ set.

        Raises:
            TypeError, ValueError:
                If X is not a valid 2D numeric array (see check_array).

        Time Complexity:
            O(n * f) for n samples and f features.

        Space Complexity:
            O(f) for the stored statistics.
        """

        X = check_array(X)

        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)
        scale = np.std(X, axis=0)

        self.scale_ = np.where(np.isclose(scale, 0), 1.0, scale)

        return self

    def transform(self, X):
        """
        Standardises X using the mean and scale learned in fit.

        Parameters:
            X:
                2D array-like with the same number of features as in fit.

        Returns:
            A new float array of shape (n_samples, n_features).

        Raises:
            NotFittedError:
                If fit has not been called.

            ValueError:
                If the number of features differs from fit.

        Time Complexity:
            O(n * f) for n samples and f features.

        Space Complexity:
            O(n * f) for the returned array.
        """

        check_is_fitted(self, ["mean_", "scale_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but StandardScaler was "
                f"fitted with {self.n_features_in_}."
            )

        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """
        Fits to X, then transforms it. Same as fit(X).transform(X).
        """

        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Scales numerical data by mapping features into a specified range, using
    the minimum and maximum of each feature.

    APIs implemented:
      - `fit(X) -> self`: stores the minimum and maximum of each feature to
         learn the range from training data.

      - `transform(X) -> X_out`: Uses the stored values from fit(X) to scale
         data into configured target range.

      - `fit_transform(X) -> X_out`: fits the transformer to X, then transforms
         using the learned state.
    """

    def __init__(self, feature_range=(0, 1)):
        """
        Parameters:
            feature_range:
                (min, max) pair of numbers for the output range.
                Defaults to (0, 1).

        Raises:
            ValueError:
                If min is not less than max.
        """

        low, high = feature_range

        if low >= high:
            raise ValueError("feature_range min must be less than max.")

        self.feature_range = (low, high)

    def fit(self, X):
        """
        Computes the per-feature minimum, maximum and range.

        Parameters:
            X:
                2D array-like of shape (n_samples, n_features).

        Returns:
            self, with data_min_, data_max_, data_range_ and
            n_features_in_ set.

        Raises:
            TypeError, ValueError:
                If X is not a valid 2D numeric array (see check_array).

        Time Complexity:
            O(n * f) for n samples and f features.

        Space Complexity:
            O(f) for the stored statistics.
        """

        X = check_array(X)

        self.n_features_in_ = X.shape[1]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        return self

    def transform(self, X):
        """
        Scales X into feature_range using the values learned in fit.

        Constant features map to the range minimum. Values outside the
        fitted min/max are not clipped, so they can fall outside
        feature_range.

        Parameters:
            X:
                2D array-like with the same number of features as in fit.

        Returns:
            A new float array of shape (n_samples, n_features).

        Raises:
            NotFittedError:
                If fit has not been called.

            ValueError:
                If the number of features differs from fit.

        Time Complexity:
            O(n * f) for n samples and f features.

        Space Complexity:
            O(n * f) for the returned array.
        """

        check_is_fitted(self, ["data_min_", "data_range_", "n_features_in_"])

        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but MinMaxScaler was "
                f"fitted with {self.n_features_in_}."
            )

        safe_range = np.where(self.data_range_ == 0, 1.0, self.data_range_)

        low, high = self.feature_range
        X_std = (X - self.data_min_) / safe_range

        return X_std * (high - low) + low

    def fit_transform(self, X):
        """
        Fits to X, then transforms it. Same as fit(X).transform(X).
        """

        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    Transforms categorical columns into indicator columns for each learned
    category.

    APIs implemented:
      - `fit(X) -> self`: learns and stores the categories for each
         categorical feature.

      - `transform(X) -> X_out`: Uses the learned categories from fit(X)
         to transform input data into indicator columns for each categorical
         feature.

      - `fit_transform(X) -> X_out`: fits the transformer to X, then
         transforms using the learned state.
    """

    def __init__(self, handle_unknown="error"):
        """
        Parameters:
            handle_unknown:
                What transform does with categories not seen in fit.
                "error" (default) raises a ValueError; "ignore" encodes
                them as all zeros.

        Raises:
            ValueError:
                If handle_unknown is not "error" or "ignore".
        """

        if handle_unknown not in ("error", "ignore"):
            raise ValueError(
                f"handle_unknown must be 'error' or 'ignore', got {handle_unknown!r}."
            )

        self.handle_unknown = handle_unknown

    def fit(self, X):
        """
        Learns the sorted unique categories of each feature.

        Parameters:
            X:
                2D array-like of shape (n_samples, n_features). Strings
                and numbers are both accepted.

        Returns:
            self, with categories_ (one array per feature) and
            n_features_in_ set.

        Raises:
            TypeError, ValueError:
                If X is not a valid 2D array (see check_array).

        Time Complexity:
            O(n log n * f) for n samples and f features (sorting).

        Space Complexity:
            O(k) where k is the total number of categories.
        """

        X = check_array(X, dtype=None)

        self.n_features_in_ = X.shape[1]

        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]

        return self

    def transform(self, X):
        """
        Encodes X as 0/1 indicator columns using the categories from fit.

        Output columns follow the order of categories_, feature by
        feature. X must hold the same kind of data as in fit (e.g. the
        string "1" and the number 1 are different categories).

        Parameters:
            X:
                2D array-like with the same number of features as in fit.

        Returns:
            A float array of shape (n_samples, total categories).

        Raises:
            NotFittedError:
                If fit has not been called.

            ValueError:
                If the number of features differs from fit, or X holds
                an unknown category and handle_unknown is "error".

        Time Complexity:
            O(n * k) for n samples and k total categories.

        Space Complexity:
            O(n * k) for the returned array.
        """

        check_is_fitted(self, ["categories_", "n_features_in_"])

        X = check_array(X, dtype=None)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OneHotEncoder was "
                f"fitted with {self.n_features_in_}."
            )

        blocks = []

        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            block = col[:, None] == cats

            unknown = ~block.any(axis=1)
            if self.handle_unknown == "error" and unknown.any():
                raise ValueError(
                    f"Unknown categories in feature {j}: {np.unique(col[unknown])}"
                )

            blocks.append(block)

        return np.hstack(blocks).astype(float)

    def fit_transform(self, X):
        """
        Fits to X, then transforms it. Same as fit(X).transform(X).
        """

        return self.fit(X).transform(X)


class SimpleImputer:
    """
    Replaces missing values (NaN) with a constant.

    APIs implemented:
      - `fit(X) -> self`: records the number of features. Nothing else is
         learned, since the fill value is a constant.

      - `transform(X) -> X_out`: replaces every NaN in X with fill_value.

      - `fit_transform(X) -> X_out`: fits the transformer to X, then
         transforms using the learned state.
    """

    def __init__(self, fill_value=0.0):
        """
        Parameters:
            fill_value:
                Number used to replace NaN. Defaults to 0.0.

        Raises:
            TypeError:
                If fill_value is not a number.
        """

        if not isinstance(fill_value, (int, float, np.number)):
            raise TypeError("fill_value must be a number.")

        self.fill_value = fill_value

    def fit(self, X):
        """
        Records the number of features in X.

        Parameters:
            X:
                2D array-like of shape (n_samples, n_features). May contain
                NaN.

        Returns:
            self, with n_features_in_ set.

        Raises:
            TypeError, ValueError:
                If X is not a valid 2D numeric array (see check_array).

        Time Complexity:
            O(n * f) for n samples and f features (input validation).

        Space Complexity:
            O(1).
        """

        X = check_array(X, allow_nan=True)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """
        Replaces every NaN in X with fill_value.

        Returns a new array; the input is not modified.

        Parameters:
            X:
                2D array-like with the same number of features as in fit.

        Returns:
            A new float array of shape (n_samples, n_features).

        Raises:
            NotFittedError:
                If fit has not been called.

            ValueError:
                If the number of features differs from fit.

        Time Complexity:
            O(n * f) for n samples and f features.

        Space Complexity:
            O(n * f) for the returned array.
        """

        check_is_fitted(self, ["n_features_in_"])

        X = check_array(X, allow_nan=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SimpleImputer was "
                f"fitted with {self.n_features_in_}."
            )

        return np.where(np.isnan(X), self.fill_value, X)

    def fit_transform(self, X):
        """
        Fits to X, then transforms it. Same as fit(X).transform(X).
        """

        return self.fit(X).transform(X)
