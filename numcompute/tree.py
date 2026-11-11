"""
This module implements a decision tree classifier that supports batch
and streaming learning.

It provides a depth-limited decision tree classifier, that is built
using only plain Python and NumPy. It also supports Gini impurity and
entropy used to select feature thresholds with configurable stopping
criterion and feature selection.

During streaming, the incoming data chunks are gathered in an internal
buffer, and once it reaches the configured size, then the tree is
retrained using the buffered sample. The predictions between the
retraining events also use the most recently trained tree.

Components:
    - _Node: Is the internal representation of decision tree nodes.
    - gini: Is what calculates the Gini impurity.
    - entropy: Is what calculates the entropy impurity.
    DecisionTreeClasifier: Supports fit(), partial_fit() and predict().

Implementation:
    - This module:
        - Uses NumPy for numerical computation and data manipulation.
        - Uses recursive tree constructions.
        - Supports deterministic split selection and class-label tie
            resolution.
        - Validates the input dimensions, feature counts and numerical
          values.
"""

import numpy as np

from .utils import check_1d_array, check_array


class _Node:
    """
    This class represent a node in the decision tree

    The internal nodes store a feature index, as well as a threshold,
    which are used for splitting. The nodes also store the references
    for their left and right child nodes. ALL leaf nodes store a
    predicted class label in value.
    """

    def __init__(self):
        self.feature = self.threshold = None
        self.left = self.right = self.value = None


def gini(y):
    """
    This function calculates the Gini impurity of a set of class
    labels.

    Parameters:
        y:
            Is a one-dimensional array of class-labels.

    Returns:
        float:
            The Gini impurity of the provided class distribution. If
            y has no samples then it simply return 0.0.
    """
    if len(y) == 0:
        return 0.0
    _, p_counts = np.unique(y, return_counts=True)
    p = p_counts / len(y)
    return 1.0 - (p**2).sum()


def entropy(y):
    """
    Calculates the entropy impurity of a set of class labels.

    This function measures the uncertainty of class distribution. A
    pure set, where all samples belong to the same class have an
    entropy of 0. Higher values indicate a more mixed class distribution.

    Parameters:
        y: One-dimensional array of class labels.

    Returns:
        float: The entropy of the class distribution, unless y contains
        zero samples which returns 0.0.
    """
    if len(y) == 0:
        return 0.0

    _, p_counts = np.unique(y, return_counts=True)
    p = p_counts / len(y)
    p = p[p > 0]

    return -np.sum(p * np.log2(p))


class DecisionTreeClassifier:
    """
    This class is a depth-limited decision tree classifier that supports
    both batch and buffered streaming learning.

    It can use either a Gini impurity or entropy, which is used to
    evaluate candidate splits. batch training is performed using fit(),
    while streamed samples are gathered in the internal buffer of
    partial_fit() and used to retraing the tree when the configred
    buffer size is reached.

    Using the most recently trained tree allows prediction to be made
    between the streaming retraining events.
    """

    def __init__(
        self,
        max_depth=3,
        criterion="gini",
        min_samples_split=2,
        max_features=None,
        buffer_size=500,
    ):
        """
        The initialisation function for the decision tree clasifier
        for batch and buffered streaming learning.

        Parameters:
            max_depth:
                The maximum depth of the tree, if the value is 0 then
                it produces a single leaf node.

            criterion:
                The impurity measure used to evaluate the candidate
                splits. Only values accepted are "gini" or "entropy".

            min_samples_split:
                The minimum number of samples that are required in
                a node so that it can be considered to be split further.

            max_features:
                The maximum number of features that are considered when
                searching for a split. All features are considered if
                None.

            buffer_size:
                The number of streamed samples that are accumulated before
                retraining the tree.

        Raises:
            TypeError:
                If the specified parameter is an invalid type.

            ValueError:
                If the specified parameter is outside the valid range.
        """

        if type(max_depth) is not int:
            raise TypeError(f"Max depth must be an integer, not {type(max_depth)}")
        if max_depth < 0:
            raise ValueError("Max depth cannot be a negative.")

        if type(criterion) is not str:
            raise TypeError(f"criterion: {type(criterion)}, must be a string")
        if criterion.lower() not in ("gini", "entropy"):
            raise ValueError(
                "The criterion must be a string, either"
                f"'gini' or 'entropy'. Got: {criterion}"
            )

        if type(min_samples_split) is not int:
            raise TypeError(
                f"min_sample_split must be an integer, not {type(min_samples_split)}"
            )
        if min_samples_split < 2:
            raise ValueError(
                f"min_samples_split: {min_samples_split} cannot be less than 2"
            )

        if max_features is not None:
            if type(max_features) is not int:
                raise TypeError(
                    "Max number of features must be an integer"
                    f" not {type(max_features)}"
                )
            elif not (max_features > 0):
                raise ValueError(
                    f"Max number of features: {max_features} cannot be"
                    " zero or a negative integer"
                )

        if type(buffer_size) is not int:
            raise TypeError(f"Buffer size must be an integer not {type(buffer_size)}")
        if buffer_size <= 0:
            raise ValueError("Buffer size must be positive.")

        self.max_depth = max_depth
        self.criterion = criterion.lower()
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size

        self.root = None
        self.n_features_in_ = None
        self._X_buffer = None
        self._y_buffer = None

    def _impurity(self, y):
        """
        This function calculates the impurity using the configured
        criterion: 'gini' or entropy'.

        It returns the Gini impurity or entropy depending on what
        self.criterion is initialised as.
        """
        if self.criterion == "gini":
            return gini(y)
        return entropy(y)

    def _leaf_value(self, y):
        """
        Returns the most frequent class label for a leaf node.

        If multiple classes are found with same maximum frequency,
        then the tie is resolved deterministically by selecting
        the smallest class label.

        Paremeters:
            y: One-dimensional ndarray of class labels for samples
               reaching the node.

        Returns:
            The class label predicted by the leaf node.
        """
        unique_classes, unique_classes_counts = np.unique(y, return_counts=True)

        max_count_pos = np.argmax(unique_classes_counts)

        return unique_classes[max_count_pos]

    # --- brute force: every feature x every midpoint ---
    def _best_split(self, X, y):
        """
        This functions find the feature and the threshold that produces
        the lowest weighted impurity.

        For every feature that is eligible, the function generates candidate
        thresholds that are the midpoints between adjacent unique feature
        values. The candidate thresholds then divides the sample into left
        or right subset, and the resulting impurity is weighted by the number
        of samples in each subset.

        A split is only selected if it improves the current unsplit node's
        impurity.

        Parameters:
            X:
                Is a two-dimensional ndarray that has shape
                (n_samples, n_features) that contain the feature
                values for the current node.

            y:
                A one-dimensional ndarray that contains the class lables
                that correspond to the rows in X.

        Returns:
            tuple: Is the index of the best features and their
            corresponding thresholds. Returns (None, None) if no
            candidate is found that improves impurity.
        """
        best_f, best_t, best_score = None, None, self._impurity(y)
        n_features = X.shape[1]
        if self.max_features is None:
            features_to_check = np.arange(n_features)
        else:
            features_to_check = np.arange(min(self.max_features, n_features))

        for f in features_to_check:
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            for t in (vals[:-1] + vals[1:]) / 2:
                mask = X[:, f] <= t
                L, R = y[mask], y[~mask]
                if len(L) == 0 or len(R) == 0:
                    continue
                score = (len(L) * self._impurity(L) + len(R) * self._impurity(R)) / len(
                    y
                )
                if score < best_score - 1e-12:
                    best_f, best_t, best_score = f, t, score
        return best_f, best_t

    # --- the recursion: this IS the training ---
    def _build(self, X, y, depth):
        """
        The main function that builds the decision tree recursively
        using the provided samples.

        The conditions required for a node to become a leaf is if:
            - the maximum tree depth is reached.
            - all the samples belong to the same class.
            - there are fewer than min_samples_split remaining
            - no candidate split if found to reduce the current impurity.

        If none of these are met, then it selects the best feature
        and threshold, before diving the data into left and right
        subsets. Then it recursively constructs the child nodes.

        Parameters:
            X:
                Is a two-dimensional ndarray that contains the feature
                values for the samples that reach the current node.

            y:
                Is a one-dimensional ndarray that contains the
                corresponding class labels.

            depth:
                The current depth of the node, with the root node
                always has depth 0.

        Returns:
            _Node:
                It returns the root node of the subtree constructed
                using the provided samples.
        """
        node = _Node()
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
        ):
            node.value = self._leaf_value(y)
            return node
        f, t = self._best_split(X, y)
        if f is None:  # nothing beat doing nothing
            node.value = self._leaf_value(y)
            return node
        mask = X[:, f] <= t
        node.feature, node.threshold = f, t
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    @staticmethod
    def _validate_train_data(X, y):
        """
        This function validates a 2D numeric feature matrix and its
        matching 1D class labels.
        """

        X = check_array(X, dtype=float)
        y = check_1d_array(y, dtype=float)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X contains {X.shape[0]} samplesbut y contains {y.shape[0]} labels."
            )

        return X, y

    def fit(self, X, y):
        """
        This functions trains the decision tree from scratch using
        the provided dataset.

        The input data is thoroughly validates before training is started.
        If fit is called, it replaces any previously trained tree and also
        clears any pending streaming data from the internal buffers.

        Parameters:
            X:
                A two-dimnesional numeric array-like that has shape
                (n_samples, n_features) and contains the training
                features.
            y:
                A one-dimensional array-like that has shape (n_samples,)
                and contains the corresponding class lables.

        Returns:
            The DecisionTreeClassifier, a fitted classifier

        Raises:
            ValueError:
                If X or y has an invalid shape, has no samples or
                features, or contains invalid values.
        """
        X, y = self._validate_train_data(X, y)

        self.root = self._build(X, y, 0)
        self.n_features_in_ = X.shape[1]

        self._X_buffer = None
        self._y_buffer = None

        return self

    def predict(self, X):
        """
        This function predicts class labels for one or more samples.

        Until a leaf node is reached, each sample is passed through
        the trained decision tree by following the feature-threshold
        comparisons.

        Parameters:
            X:
                Can be a numeric array-like that has shape
                (n_samples, n_features) or a one-dimensional array
                that represents a single sample.

        Returns:
            ndarray:
                A one-dimensional array that contains the predicted
                class label for each of the input samples.

        Raises:
            AttributeError:
                If the classifier hasn't been fitted yet.

            TypeError:
                If X cannot be converted to numeric data.

            ValueError:
                If X has invalid shape, has no samples, has a different
                number of features compared to the fitted model or has
                NaN or infinite values.
        """
        if self.root is None:
            raise AttributeError("The classifier has not been fitted yet")
        try:
            X = np.atleast_2d(np.asarray(X, dtype=float))
        except (TypeError, ValueError) as conv_err:
            raise TypeError("X must only contain numeric values") from conv_err

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = check_array(X, dtype=float)

        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but the classifier was fitted"
                f" with {self.n_features_in_} features."
            )

        preds = []
        for x in X:
            node = self.root
            while node.value is None:  # while it is a question, not a leaf
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            preds.append(node.value)
        return np.array(preds)

    def partial_fit(self, partial_X, partial_y):
        """
        This function updates the classifier using a part of streaming
        data.

        Any incoming samples are appended to an internal buffer, and
        once the number of buffered samples reaches or exceeds the
        buffer_size, then the decision tree is retrained from scratch
        using all of the buffered samples. The buffers are also cleared.

        Predictions continue to use the most recently trained tree in
        between the retraining events.

        Parameters:
            partial_X:
                Is a two-dimensional numeric array-like that has shape
                (n_samples, n_features), that contain the streamed
                feature values.

            partial_y:
                Is a one-dimensional array-like that has shape
                (n_samples,) with the corresponding class labels.

        Returns:
            DecisionTreeClassifier:
                It returns the updated classifier.

        Raises:
            TypeError:
                Is raised if partial_X cannot be converted into a
                numeric feature data.

            ValueError:
                Is raised if the partial data has invalid dimensions,
                has no samples or features, has inconsistent sample
                counts, has invalid values, or has different number
                of features from the previous chunks or fitted data.
        """
        partial_X, partial_y = self._validate_train_data(partial_X, partial_y)

        if self.n_features_in_ is None:
            self.n_features_in_ = partial_X.shape[1]

        elif self.n_features_in_ != partial_X.shape[1]:
            raise ValueError(
                "The current chunk must have the same number of features"
                " as the existing model data."
            )

        if self._X_buffer is None:
            self._X_buffer = partial_X.copy()
            self._y_buffer = partial_y.copy()
        else:
            self._X_buffer = np.concatenate((self._X_buffer, partial_X), axis=0)

            self._y_buffer = np.concatenate((self._y_buffer, partial_y), axis=0)

        if self._X_buffer.shape[0] >= self.buffer_size:
            train_X = self._X_buffer
            train_y = self._y_buffer

            self.fit(train_X, train_y)

        return self
