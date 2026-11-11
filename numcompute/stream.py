import sys

from .metrics import Accuracy


class StreamTrainer:
    """
    This class is responsible for managing the model training,
    evaluation and logging of the streaming machine-learning
    workflows.

    New chunks of data are passed to the model using partial_fit()
    and then score_chunks() evaluates the predictions on a chunk and
    records both the accuracy of current chunk and the cumulative
    accuracy across all scored chunks.

    An approximate memory footprint of the model is also recorded for
    each scored chunk.

    Attributes:
        model:
            The model or pipeline being trained

        metric:
            The cumulative evaluation metric.

        logs:
            A list that contains the evaluation information recorded
            for each scored chunk.

        seen_chunks_:
            The number of chunks passed to fit_chunk().
    """

    def __init__(self, model, metric=None):
        """
        Initialises the streaming traning

        Parameters:
            model:
                A model or a pipeline that implements partial_fit() and
                predict.

            metric:
                Is the optional cumulative metric that implements update(),
                result() and reset(). The default is set to Accuracy.

        Time Complexity:
            O(1)

        Space Complexity:
            O(1), not including the supplied model and metric.
        """
        self.model = model

        if metric is None:
            metric = Accuracy()

        self.metric = metric
        self.logs = []
        self.seen_chunks_ = 0

    def fit_chunk(self, X, y):
        """
        This function incrementally trains the model using a new
        chunk of data.

        Parameters:
            X:
                Feature data for the current chunk.

            y:
                The traget labels for the current chunk.

        Returns:
            StreamTrainer:
                The updated trainer/

        Time Complexity:
            Is dependant on the partial_fit() complexity of the model.

        Space Complexity:
            Is dependant on the incremental state maintained by the model.
        """
        self.model.partial_fit(X, y)
        self.seen_chunks_ += 1
        return self

    def score_chunk(self, X, y):
        """
        This function evaluates the model using the new chunk and
        a records the streaming metrics.

        The predictions are generated from the current chunk while
        accuracy is calculated seperately. The configured cumulative
        metric is updated using all the scored observations.

        The resultant chunk accuracy, cumulative metric  value
        and approximate model memory usage are then stored in
        the trainer log.

        Parameters:
            X:
                The feature data of the chunk being evaluated.

            y:
                The true labels for the chunk being evaluated.
        Returns:
            A dict, that contains the chunk numbers, accuracy, its
            cumulative accuracy and the approximate memory usage of
            the model.

        Time Complexity:
            Dependant on the model's predict complexity, in addition to
            O(n) for the metric calculations.

        Space Complexity:
            Is dependant on the prediction output with O(n) space for
            metric evaluation.
        """
        y_pred = self.model.predict(X)

        chunk_metric = Accuracy()
        chunk_metric.update(y, y_pred)
        chunk_acc = chunk_metric.result()

        self.metric.update(y, y_pred)
        cum_acc = self.metric.result()

        log = {
            "chunk": self.seen_chunks_,
            "chunk_acc": chunk_acc,
            "cum_acc": cum_acc,
            "memory_usage": self._memory_usage(),
        }

        self.logs.append(log)

        return log

    def _memory_usage(self):
        """
        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        return sys.getsizeof(self.model)

    def get_logs(self):
        """
        This function returns all the streaming evaluation logs it
        has recorded.

        Returns:
            A list that has the per-chun evaluation logs.

        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        return self.logs

    def reset(self):
        """
        Used to clear the trainer's logs, and wipe the cumulative metric
        state and chunk count.
        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        self.logs = []
        self.metric.reset()
        self.seen_chunks_ = 0
