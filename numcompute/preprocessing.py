"""
Transforms pre-loaded arrays into data suitable for analysis and streaming
machine-learning workflows.

This module provides transformations that prepare both numerical and
categorical data for further analysis or modelling. Components support
both batch learning through fit() and incremental learning through
partial_fit(), allowing the learned statistics or categories to be
updated as new chunks of data arrive.

transform(X) applies the learned state to new data without
relearning it. NumPy vectorisation and broadcasting are used to implement
all numerical operations.

Classes Included:
- `StandardScaler` (z-score standardisation)
- `MinMaxScaler` (scale to a given range)
- `OneHotEncoder` (for categorical variables)
- `SimpleImputer` (replace NaN using learned per-feature means)

Implementation requirements:
- Core transformations should be implemented using vectorised NumPy operations,
not Python loops where possible.
- Streaming updates preserve previously learned state without storing all
previously processed data.
"""

import numpy as np

from numcompute.utils import check_array, check_is_fitted


class StandardScaler:
    """
    Standardises numerical features, centres them by subtracting their means
    and scales them by dividing the difference by their standard deviation.

    The scaler supports both batch and streaming learning, where fit()
    learns the feature statistics from scratch and partial_fit() updates
    the running mean and variance as new chunks of data arrive.

    APIs implemented:
    - `fit(X) -> self`: calculates and stores the mean and standard
        deviation per feature from scratch.

    - `partial_fit(X) -> self`: incrementally updates the running mean
      and variance using new chunk of data.

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
            StandardScaler:
                The fitted scaler is returned with mean_, scale_, M2_,
                n_samples_seen_ and n_features_in_ set.

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
        self.n_samples_seen_ = X.shape[0]

        self.mean_ = np.mean(X, axis=0)

        self.M2_ = np.sum((X - self.mean_) ** 2, axis=0)

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis = 0)

        var = self.M2_ / self.n_samples_seen_
        scale = np.sqrt(var)
        
        constant_features = self.data_min_ == self.data_max_

        self.scale_ = np.where(constant_features, 1.0, scale)

        return self

    def partial_fit(self, X):
        """
        This function incrementally updates the scaler's mean and
        variance using a new chunk of data.

        It also preserves statistics from previous chunks and combines
        them with the new chunk of data through numerically stable
        running calculations.

        Parameters:
            X:
                This is a two-dimenional numeric array-like parameter
                of shape (n_samples, n_features).
        Returns:
            StandardScaler:
                The updated scaler.
        Raises:
            TypeError, ValueError:
                If X is invalid, or its feature count is different
                from previously processed chunks.

        Time Complexity:
            O(n * f) where n is the number of samples present in the
            current chunk and f is the number of features.

        Space Complexity:
            O(n * f) temporary space for intermediate array operations,
            with O(f) persistent space for running mean, M2, scale,
            and feature statistics.
        """
        X = check_array(X)

        n_chunk = X.shape[0]
        n_features = X.shape[1]

        mean_chunk = np.mean(X, axis=0)
        chunk_m2 = np.sum((X - mean_chunk) ** 2, axis=0)

        min_chunk = np.min(X, axis=0)
        max_chunk = np.max(X, axis=0)

        if not hasattr(self, "n_samples_seen_"):
            self.n_features_in_ = n_features
            self.n_samples_seen_ = n_chunk
            self.mean_ = mean_chunk
            self.M2_ = chunk_m2
            self.data_max_ = max_chunk
            self.data_min_ = min_chunk

        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError("X must have same number of features.")
            old_count = self.n_samples_seen_
            total_count = old_count + n_chunk

            delta = mean_chunk - self.mean_
            new_mean = self.mean_ + delta * (n_chunk / total_count)

            self.M2_ = (
                self.M2_ + chunk_m2 + (delta**2) * old_count * n_chunk / total_count
            )

            self.n_samples_seen_ += n_chunk
            self.mean_ = new_mean

            self.data_min_ = min_chunk
            self.data_max_ = max_chunk

        var = self.M2_ / self.n_samples_seen_
        scale = np.sqrt(var)

        constant_features = self.data_min_ == self.data_max_

        self.scale_ = np.where(constant_features, 1.0, scale)

        return self

    def transform(self, X):
        """
        Standardises X using currently learned mean and scale.

        The statistic may have been learned using fit() or
        updated incrementally using partial_fit().

        Parameters:
            X:
                2D array-like with the same number of features as in fit.

        Returns:
            A new float array of shape (n_samples, n_features).

        Raises:
            NotFittedError:
                If the scaler has not been fitted or incrementally updated.

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

    The scaler supports both batch and streaming learning where fit() learns
    the feature ranges from scratch and partial_fit() expands the running min
    and max values as new chunks of data arrive.

    APIs implemented:
      - `fit(X) -> self`: learns the minimum and maximum and range of each
        feature from scratch.

      - `partial_fit(X) -> self`: incrementally updates the running min
        and max values.

      - `transform(X) -> X_out`: scales data into the configured target range
         using the currently learned statistics.

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

    def partial_fit(self, X):
        """
        This function incrementally updates the per-feature min and
        max values using a new chunk of data.

        For the first chunk, the observed min and max values are stored
        directly, then later on they are combined with previously learned
        values to maintain min and max across all data thus processed.

        Parameters:
            X:
                This is a two-dimenional numeric array-like of shape
                (n_samples, n_features).

        Returns:
            MinMaxScaler:
                The updated scaler.
        Raises:
            TypeError, ValueError:
                If X is invalid, or its feature count is different
                from previously processed chunks.

        Time Complexity:
            O(n * f) where n is the number of samples in the current
            chunk and f is the number of features.

        Space Complexity:
            O(f) persistent space for running min, max and range
            statistics.
        """
        X = check_array(X)

        n_features = X.shape[1]

        chunk_min = np.min(X, axis=0)
        chunk_max = np.max(X, axis=0)

        if not hasattr(self, "data_min_"):
            self.n_features_in_ = n_features
            self.data_min_ = chunk_min
            self.data_max_ = chunk_max
        else:
            if n_features != self.n_features_in_:
                raise ValueError("The number of features must match.")

            self.data_min_ = np.minimum(self.data_min_, chunk_min)
            self.data_max_ = np.maximum(self.data_max_, chunk_max)

        self.data_range_ = self.data_max_ - self.data_min_

        return self

    def transform(self, X):
        """
        Scales X into feature_range using currently learned min and max
        values.

        The statistic may have been learned using fit() or
        updated incrementally using partial_fit().

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
                If the scaler has not been fitted or incrementally updated.

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

    The encoder supports both batch and streaming learning with fit()
    learning the categorises from scratch while partial_fit() expands the
    stored categories whenever the previously unseen values show up in
    new chunks of data.

    APIs implemented:
      - `fit(X) -> self`: learns the unique categories for each
         categorical feature.

      - `partial_fit(X) -> self`: incrementally adds the categories observed
        into new chunks.

      - `transform(X) -> X_out`: encodes data into indicator columns using the
        currently learned categories.

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
        Encodes X as 0/1 indicator columns using currently learned
        categories.

        The statistic may have been learned using fit() or
        updated incrementally using partial_fit().

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
                    If the encoder has not been fitted or incrementally
                    updated.

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

    def partial_fit(self, X):
        """
        This function incrementally updates the categories learned for each
        feature.

        For the first chunk, the observed categories in the new chunk
        are combined with categories learned from previous data. All
        duplicate values are removed and the resulting category arrays
        remain sorted.

        Parameters:
            X:
                This is a two-dimenional array-like of shape
                (n_samples, n_features). Numeric or categorical
                values may be provided.

        Returns:
            OneHotEncoder:
                The updated encoder.

        Raises:
            TypeError, ValueError:
                If X is invalid, or its feature count is different
                from previously processed chunks.

        Time Complexity:
            O(n log n * f) where n is the number of samples in the current
            chunk and f is the number of features.

        Space Complexity:
            O(k) where k is the total number of categories currently
            stored across all features.
        """
        X = check_array(X, dtype=None)

        n_features = X.shape[1]

        if not hasattr(self, "categories_"):
            self.n_features_in_ = n_features
            self.categories_ = [np.unique(X[:, j]) for j in range(n_features)]
        else:
            if n_features != self.n_features_in_:
                raise ValueError(
                    "X must have same number of features aspreviously processed data."
                )
            for feature in range(n_features):
                new_categories = np.unique(X[:, feature])
                self.categories_[feature] = np.unique(
                    np.concatenate((self.categories_[feature], new_categories), axis=0)
                )

        return self

    def fit_transform(self, X):
        """
        Fits to X, then transforms it. Same as fit(X).transform(X).
        """

        return self.fit(X).transform(X)


class SimpleImputer:
    """
    Replaces missing numeric values with learned per-feature means.

    The imputer supports batch and streaming learning, where fit() learns
    the imputation statistics from scratch, and partial_fit() updates the
    running statistics as new chunks of data arrive. fill_value is used as
    a fallback if a feature never had a valid value.

    APIs implemented:
      - `fit(X) -> self`: learns the per feature mean imputation values.
      - `partial_fit(X) -> self`: incrementally updates the learned means.
      - `transform(X) -> X_out`: replaces every NaN with learned statistics.
      - `fit_transform(X) -> X_out`: fits the transformer to X, then
         transforms using the learned state.
    """

    def __init__(self, fill_value=0.0):
        """
        Parameters:
            fill_value:
               Fallback value that is used for a feature if there are no
               observed non-missing values. Defaults to 0.0.

        Raises:
            TypeError:
                If fill_value is not a number.
        """

        if not isinstance(fill_value, (int, float, np.number)):
            raise TypeError("fill_value must be a number.")

        self.fill_value = fill_value

    def fit(self, X):
        """
        Fits the imputer from scratch using the provided data.

        All previously learned statistics are discarded and the mean of
        the observed non-missing values in each feature is learned and
        stored as the imputation statistic. If a feature only has missing
        values then fill_value is used as a fallback statistic.

        Parameters:
            X:
                2D array-like of shape (n_samples, n_features). Missing
                values represented by NaN.

        Returns:
            SimpleImputer:
                The fitted imputer.

        Raises:
            TypeError, ValueError:
                If X is not a valid 2D numeric array (see check_array).

        Time Complexity:
            O(n * f) for n samples and f features (input validation).

        Space Complexity:
            O(n * f) temporary space for n samples and f features
            (input validation).
        """
        for attribute in (
            "n_features_in_",
            "_valid_counts",
            "_valid_sums",
            "statistics_",
        ):
            if hasattr(self, attribute):
                delattr(self, attribute)

        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        This function updates the imputation statistics using a new
        chunk of data incrementally.

        Any missing values are ignored when calculating statistics and
        for each feature, the running sum and number of observed values
        are updated, and then used to calculate the current mean
        imputation value.

        Parameters:
            X:
                Is a two-dimensional numeric array-like that has shape
                (n_samples, n_features). The missing values can be
                represented by NaN.

        Returns:
            SimpleImputer:
                The updated imputer

        Raises:
            TypeError, ValueError:
                If X is not a valid 2D array or if it has a different
                number of features from the previously processed data.

        Time Complexity:
            O(n * f) where n is the number of samples present in the
            current chunk and f is the number of features.

        Space Complexity:
            O(n * f) temporary space complexity for the missing-value
            mask and also has O(f) persistent space for running sums,
            counts and imputation statistics.
        """
        X = check_array(X, allow_nan=True)
        n_features = X.shape[1]
        mask_valid = ~np.isnan(X)

        valid_chunk_count = np.sum(mask_valid, axis=0)

        chunk_sum = np.nansum(X, axis=0)

        if not hasattr(self, "_valid_counts"):
            self.n_features_in_ = n_features
            self._valid_counts = valid_chunk_count
            self._valid_sums = chunk_sum
        else:
            if n_features != self.n_features_in_:
                raise ValueError("The number of features must match.")

            self._valid_counts += valid_chunk_count
            self._valid_sums += chunk_sum

        self.statistics_ = np.full(self.n_features_in_, self.fill_value, dtype=float)

        np.divide(
            self._valid_sums,
            self._valid_counts,
            out=self.statistics_,
            where=self._valid_counts > 0,
        )

        return self

    def transform(self, X):
        """
        Replaces missing values in X with learned per-feature
        imputation statistics.

        Each NaN is replaced by the running mean that is learned
        for its feature and if a feature never had a valid value
        then that feature's configured fill_value is used as a
        fallback statistic.

        Returns a new array; the input is not modified.

        Parameters:
            X:
                2D array-like with the same number of features as in fit.
                NaN values are allowed.

        Returns:
            A new float array of shape (n_samples, n_features) with missing
            values replaced.

        Raises:
            NotFittedError:
                If the imputer has not been fitted or incrementally updated.

            TypeError, ValueError:
                If the number of features differs from fit or if X is invalid.

        Time Complexity:
            O(n * f) for n samples and f features.

        Space Complexity:
            O(n * f) for the returned array.
        """

        check_is_fitted(self, ["statistics_", "n_features_in_"])

        X = check_array(X, allow_nan=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SimpleImputer was "
                f"fitted with {self.n_features_in_}."
            )

        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X):
        """
        Fits to X, then transforms it. Same as fit(X).transform(X).
        """

        return self.fit(X).transform(X)
