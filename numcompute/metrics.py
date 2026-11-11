"""
Evaluates the predicted values against known targets using classification
and regression metrics.

For classification, this module will evaluate the predicted values using
performance metrics like:

  - Accuracy, which calculates the proportion of predictions that are
    correct.

  - Precision, which measures the proportion of predicted positive cases
    that were actually positive.

  - Recall, which measures the proportion of actual positive cases that
    the model correctly identified.

  - F1 score, which combines precision and recall to evaluate the model
    on a range of [0,1].

  - Confusion matrix, which summarises the number of correct and
    incorrect predictions for each class.

  - ROC AUC, which evaluates the effectiveness of prediction scores in
    ranking the positive observations above negative observations.

For regression, this module evaluates the numerical predictions using:
  - The Mean Squared Error (MSE), which evaluates the model by calculating
    the average squared difference between the predicted values and the target
    values.


Metrics in this module are designed to operate on NumPy arrays and provide
consistent behaviour during comparison of predicted values and their related
target values.
"""

from collections import deque

import numpy as np

from .utils import check_1d_array


def _prepare_inputs(y_true, y_pred):
    """
    This helper function validates and prepars the true and predicted values
    for metric calculation.

    First, both input as converted to one-dimensional numeric NumPy arrays,
    where they must have the same number of samples. Positions that contain
    NaN are removed to leave only valid paired observations.

    Paremeters:
      y_true:
        A one-dimensional numeric array-like that contains the true values.

      y_pred:
        A one-dimensional numeric array-like containing predicted values.

    Returns:
      tuple:
        The filtered y_true and y_pred arrays that only contain
        the paired non-missing observations.

    Raises:
      TypeError, ValueError:
        Raised if either array is invalid, or they have different number
        of samples.

    Time Complexity:
      Has O(n) time complexity where n is the number of input observations.

    Space Complexity:
      O(n) for the validity mask and filtered arrays.
    """
    y_true = check_1d_array(y_true, dtype=float, allow_nan=True)
    y_pred = check_1d_array(y_pred, dtype=float, allow_nan=True)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must contain the same number of samples.")

    valid_output = ~np.isnan(y_true) & ~np.isnan(y_pred)

    return y_true[valid_output], y_pred[valid_output]


class Accuracy:
    """
    Calculates the classification accuracy across the streamed chunks of
    predictions.

    A cumulative number of the correct number of predictions, and total valid
    observations is maintained without storing previously processed prediction
    arrays.

    Methods:
      update(y_true, y_pred):
        It updates the cumulative metric using the new chunk of data.

      result():
        Returns the cumulative accuracy.

      reset():
        Resets the metric state.
    """

    def __init__(self):
        """
        Initialises the class

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """
        self.reset()

    def update(self, y_true, y_pred):
        """
        Updates the cumulative accuracy using new chunk of predictions.

        Parameters:
          y_true:
            A one-dimensional numeric array-like that contains true labels.

          y_pred:
            A one-dimensional numeric array-like that contains the predicted
            labels.

        Returns:
          Accuracy:
            The updated metric.

        Raises:
          TypeError, ValueError:
            If any of the inputs are invalid or if the array contains different
            number of samples.

        Time Complexity:
          O(n) where n is the number of predictions in the current chunk.

        Space Complexity:
          Has O(n) temporary space for the validated and filtered input arrays,
          and has O(1) persistent metric state.
        """
        y_true, y_pred = _prepare_inputs(y_true, y_pred)
        chunk_correct = np.sum(y_pred == y_true)

        self.correct_ += chunk_correct
        self.total_ += y_true.size

        return self

    def result(self):
        """
        This function returns the current cumulative accuracy.

        Returns:
          float:
            The proportion of valid prediction that are correct or 0.0 if
            no valid observation is processed.

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """

        if self.total_ == 0:
            return 0.0
        return self.correct_ / self.total_

    def reset(self):
        """
        Resets the accumulated accuracy state.

        Returns:
          None

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """
        self.correct_ = 0
        self.total_ = 0


class _BinaryCounts:
    """
    Maintains cumulative binary-classification counts across streamed
    chunks of predictions.

    Counts of true-positive, false-positive, false-negative and true-negative
    are updated incrementally for the configured positive class. Prediction
    arrays processed previously are not stored.

    These cumulative counts are used by subclasses to calculate their own metric
    results.

    """

    def __init__(self, prediction_class=1):
        """
        This intialises the cumulative binary classification counts.

        Parameters:
          prediction_class:
            The class label treated as the positive class, defaulted to 1.

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """
        self.prediction_class = prediction_class
        self.reset()

    def update(self, y_true, y_pred):
        """
        This updates the cumulative binary classification counts using the
        new chunk of data.

        The counts accumulated for the configured positive class are
        true-positive, false-positive, false-negative and true-negative.

        Parameters:
          y_true:
            A one-dimensional numeric array-like containing true labels.

          y_pred:
            One-dimensional numeric array-like containing predicted
            labels.

        Returns:
          _BinaryCounts:
            The updated metric instance.

        Raises:
          TypeError, ValueError:
            If any of the inputs are invalid or if the array contains different
            number of samples.

        Time Complexity:
          O(n) where n is the number of predictions in the current chunk.

        Space Complexity:
          Has O(n) temporary space for the validated and filtered input arrays,
          and has O(1) persistent metric state.
        """
        y_true, y_pred = _prepare_inputs(y_true, y_pred)

        pred_pos = y_pred == self.prediction_class
        pred_neg = y_pred != self.prediction_class
        true_pos = y_true == self.prediction_class
        true_neg = y_true != self.prediction_class

        true_pos_chunk = np.sum(pred_pos & true_pos)
        false_pos_chunk = np.sum(pred_pos & true_neg)
        false_neg_chunk = np.sum(pred_neg & true_pos)
        true_neg_chunk = np.sum(pred_neg & true_neg)

        self.true_pos_ += true_pos_chunk
        self.false_pos_ += false_pos_chunk
        self.false_neg_ += false_neg_chunk
        self.true_neg_ += true_neg_chunk

        return self

    def result(self):
        """
        Returns the result of the metric calculated by the subclass.

        Raises:
          NotImplementedError:
            Because the subclass has to provide its own version of
            the result calculation.
        """
        raise NotImplementedError("Subclasses implement result()")

    def reset(self):
        """
        Resets the accumulated binary classification counts.

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """
        self.true_pos_ = 0
        self.false_pos_ = 0
        self.false_neg_ = 0
        self.true_neg_ = 0


class Precision(_BinaryCounts):
    """
    This class incrementally calculates the precision for the configured
    positive class.
    """

    def result(self):
        """
        Returns the current cumulative precision.

        Returns:
          float:
            Precision for the configured positive class, and returns 0.0
            if no positive prediction have been observed.

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """
        if (self.true_pos_ + self.false_pos_) == 0:
            return 0.0

        return self.true_pos_ / (self.true_pos_ + self.false_pos_)


class Recall(_BinaryCounts):
    """
    This class incrementally calculates the recall for the configured
    positive class.
    """

    def result(self):
        """
        This function returns the current cumulative recall.

        Returns:
          float:
            Recall for the configured positive class, and 0.0 if there
            are no observed positive predictions.

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """
        if (self.true_pos_ + self.false_neg_) == 0:
            return 0.0

        return self.true_pos_ / (self.true_pos_ + self.false_neg_)


class F1(_BinaryCounts):
    """
    This class incrementally calculates the F1 score for the configured
    positive class.
    """

    def result(self):
        """
        This function returns the cumulative F1 score.

        Returns:
          floatL
            F1 score for the configured positive class and return 0.0
            when the score cannot be calculated from the counts.

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """
        if (self.true_pos_ + self.false_pos_ + self.false_neg_) == 0:
            return 0.0

        return (2 * self.true_pos_) / (
            (2 * self.true_pos_) + self.false_pos_ + self.false_neg_
        )


class ConfusionMatrix(_BinaryCounts):
    """
    Maintains a cumulative binary confusion matrix across the streamed chunks.

    result() returns the counts in the specified order:
      - (true_positive, false_positive, true_negative, false_negative).

      Time Complexity:
        O(1).

      Space Complexity:
        O(1).
    """

    def result(self):
        return (self.true_pos_, self.false_pos_, self.true_neg_, self.false_neg_)


class MSE:
    """
    Calculates the mean squared error across the streamed chunks.

    The metric accumulates the total squared error and number of valid
    observations without storing and predictions arrays processed previously.
    Methods:
      update(y_true, y_pred):
        It updates the cumulative squared error using the new chunk of data.

      result():
        Returns the cumulative mean squared error.

      reset():
        Resets the metric state.
    """

    def __init__(self):
        """
        Initialises the class

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """
        self.reset()

    def update(self, y_true, y_pred):
        """
        Updates the cumulative mean squared error using new chunk of
        predictions.

        Parameters:
          y_true:
            A one-dimensional numeric array-like that contains true labels.

          y_pred:
            A one-dimensional numeric array-like that contains the predicted
            labels.

        Returns:
          MSE:
            The updated metric.

        Raises:
          TypeError, ValueError:
            If any of the inputs are invalid or if the array contains different
            number of samples.

        Time Complexity:
          O(n) where n is the number of predictions in the current chunk.

        Space Complexity:
          Has O(n) temporary space for the validated and filtered input arrays,
          and has O(1) persistent metric state.
        """
        y_true, y_pred = _prepare_inputs(y_true, y_pred)
        squared_error = np.sum((y_pred - y_true) ** 2)

        self.squared_error_ += squared_error
        self.total_ += y_true.size

        return self

    def result(self):
        """
        This function returns the current cumulative mean squared error.

        Returns:
          float:
            The mean squared error across all valid observations or 0.0 if
            no valid observation is processed.

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """

        if self.total_ == 0:
            return 0.0
        return self.squared_error_ / self.total_

    def reset(self):
        """
          Resets the accumulated mean squared error state state.

          Returns:
            None

        Time Complexity:
            O(1).

          Space Complexity:
            O(1).
        """
        self.squared_error_ = 0
        self.total_ = 0


class AUC:
    """
    It incrementally accumulates the prediction scores for calculating
    the binary ROC area under the curve.

    The valid true labels and prediction scored from the newly streamed
    chunks are stored in every update. result() calculates the accumulative
    AUC using all of the observations that have been processed since the last
    reset.

    Methods:
      update(y_true, y_score):
        It adds a new chunk of labels and predition scores.

      result():
        It return the resulting ROC AUC

      reset():
        It resets the accumulated labels and scores.
    """

    def __init__(self, prediction_class=1):
        """
        Initalises the class

        Time Complexity:
          O(1).

        Space Complexity:
          O(1).
        """
        self.prediction_class = prediction_class
        self.reset()

    def update(self, y_true, y_score):
        """
        This adds a new chunk of true labels and prediction scores

        Parameters:
          y_true:
            One-dimensional numeric array-like that contains true labels.

          y_score:
            A one-dimensional numeric array-like that contains predictions
            scores for the positive class.

        Returns:
          AUC:
            The updated metric

        Time Complexity:
          O(n).

        Space Complexity:
          O(n)
        """
        y_true, y_score = _prepare_inputs(y_true, y_score)

        self.true_y_chunks_.append(y_true.copy())
        self.score_y_chunks_.append(y_score.copy())

        return self

    def result(self):
        """
        This function calculates the cumulative ROC AUC from all the observations
        since last reset.

        All predicition scores are ranked and the tied scores are assigned their
        average ranks.The rank sum of the configured positive class is then used
        to calculate the AUC/

        Returns:
          float:
            The cumulative ROC AUC, where 0.0 is returned if only one class is
            present or no observaations are observed.

        Time Complexity:
          O(n log n) where n is the total number of observations since last reset,
          because of ranking te prediction scores.

        Space Complexity:
          O(n) because of the accumulated labels, scored, and the temporary ranking
          arrays.
        """
        if len(self.true_y_chunks_) == 0:
            return 0.0

        y_true = np.concatenate(self.true_y_chunks_)
        y_score = np.concatenate(self.score_y_chunks_)

        mask_pos = y_true == self.prediction_class
        mask_neg = ~mask_pos

        n_pos = np.sum(mask_pos)
        n_neg = np.sum(mask_neg)

        if n_pos == 0 or n_neg == 0:
            return 0.0

        _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)

        cum_counts = np.cumsum(counts)

        start_ranks = cum_counts - counts + 1
        end_ranks = cum_counts

        avg_rank = (start_ranks + end_ranks) / 2

        ranks = avg_rank[inv]

        sum_pos_rank = np.sum(ranks[mask_pos])

        return (sum_pos_rank - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    def reset(self):
        """
        Resets all the labels and prediction scores.

        Time Complexity:
          O(n) where n is the accumulated number of observations being
          released.

        Space Complexity:
          O(1) new persistent state.
        """
        self.true_y_chunks_ = []
        self.score_y_chunks_ = []


class RollingWindowMetric:
    """
    This class is used to calculate the metric using recent observations
    from the streaming data sequence.

    Antoher metric implementing the methods: update(), result() and reset()
    is wrapped by this class, and observations are maintained in a fixed-size
    window. If the window reaches it max size then the oldest observations are
    automatically replaced by their newer counterparts.

    Parameters:
      metric:
        A metric object that implements update(), result(), and reset().

      window_size():
        Sets the max number of recent observations that are valid in the
        rolling window, the default is 100.

    Methods:
      update(y_true, y_pred):
        Adds new chunk of observation to the rolling window.

      result():
        Calculates the wrapped metric, only uses the observations currently
        in the window.

      reset():
        Resets/clears the rolling window and wrapped metric.
    """

    def __init__(self, metric, window_size=100):
        """
        Initialises and validates the class.
        Time Complexity:
          O(1)

        Space Complexity:
          O(1)
        """
        if not isinstance(window_size, int):
            raise TypeError("window_size must be a int.")
        if window_size <= 0:
            raise ValueError("window_size must be a positive int greater than 0.")

        for method in ("update", "result", "reset"):
            if not callable(getattr(metric, method, None)):
                raise TypeError(
                    "The metric must implement the methods:"
                    " update(), result(), and reset()"
                )

        self.metric = metric
        self.window_size = window_size

        self.reset()

    def update(self, y_true, y_pred):
        """
        Adds new chunk of observations to the rolling window.

        Invalid pairs with NaN will be removed before addition.

        Parameters:
          y_true:
            One-dimensional numeric array-like that contains true
            values or labels.

          y_pred:
            One-dimensional numeric array-like that contains prediction
            values or labels.

        Returns:
          RollingWindowMetric:
            The updated metric

        Raises:
          TypeError, ValueError:
            If either input is invalid or their sample counts are different.

        Time Complexity:
          O(n)

        Space Complexity:
          O(w) where w is the window_size
        """

        y_true, y_pred = _prepare_inputs(y_true, y_pred)

        for true_val, pred_val in zip(y_true, y_pred, strict=False):
            self.true_window_y_.append(true_val)
            self.pred_window_y_.append(pred_val)

        return self

    def result(self):
        """
        Calculates the wrapped metric, observations in current window are used
        Returns:
          The result produced by wrapped metric.

        Time Complexity:
          O(w) for all count-based metrics and MSE, where w is number of
          observations in the current window. O(w log w) for AUC due to
          ranked predictions.

        Space Complexity:
          O(w) for temporary NumPy arrays.
        """
        self.metric.reset()

        if len(self.true_window_y_) == 0:
            return self.metric.result()

        y_true = np.asarray(self.true_window_y_, dtype=float)
        y_pred = np.asarray(self.pred_window_y_, dtype=float)

        self.metric.update(y_true, y_pred)

        return self.metric.result()

    def reset(self):
        """
        This function clears all the observations from the rolling window
        and resets them.

        Returns:
          None

        Time Complexity:
          O(w)

        Space Complexity:
          O(1)
        """
        self.true_window_y_ = deque(maxlen=self.window_size)

        self.pred_window_y_ = deque(maxlen=self.window_size)

        self.metric.reset()
