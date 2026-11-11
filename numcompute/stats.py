"""
Provides descriptive statistics for the numerical arrays for analysis
components.

This module also provides both batch and streaming descriptive statistics.

The standalone functions calculate the statistics directly from the supplied
NumPy array, while StreamingStats, instead updates them incrementally across
chunks of data using update_stats. The cumulative statistics are maintained
over the observations processed so far.

The standalone calculates basic statistics such as mean, median, standard deviation,
min and max. It also groups observations into bins, counts how many
observations fall into each interval and also uses quantiles to identify
values that divide ordered observations into specified proportions.

The basic statistics must follow axis-wise statistics with clear dimension
and shape behaviour, and the quantiles must also handle NaN values
appropriately.
"""

import numpy as np

from .utils import check_array


def mean(data, axis=0):
    """
    Calculates the stastical mean for the inputed data

    Parameters:
        data: a numpy array containing the data to calculate the mean of
        axis: the axis to operate on, defaults to 0

    Returns:
        A scalar or an array containing the resulting means

    Raises:

    Time Complexity:
        O(n) where n is the number of elements of the data

    Space Complexity:
        O(n) where n is the number of elements of the data
    """
    return np.nanmean(data, axis)


def median(data, axis=0):
    """
    Calculates the stastical median for the inputed data

    Parameters:
        data: a numpy array containing the data to calculate the median of
        axis: the axis to operate on, defaults to 0


    Returns:
        A scalar or an array containing the resulting median

    Raises:

    Time Complexity:
        O(N * f) or greater, depends on the median operation,
        where N is the total number of observations.

    Space Complexity:
        O(N * f), stored chunk concatenated.
    """
    return np.nanmedian(data, axis)


def stan_dev(data, axis=0):
    """
    Calculates the stastical standard deviation for the inputed data

    Parameters:
        data: a numpy array containing the data to calculate the SD of
        axis: the axis to operate on, defaults to 0

    Returns:
        A scalar or an array containing the resulting SD

    Raises:

    Time Complexity:
        O(n) where n is the number of elements of the data

    Space Complexity:
        O(n) where n is the number of elements of the data
    """
    return np.nanstd(data, axis)


def min(data, axis=0):
    """
    Calculates the min for the inputed data

    Parameters:
        data: a numpy array containing the data to calculate the min of
        axis: the axis to operate on, defaults to 0


    Returns:
        A scalar or an array containing the resulting mins

    Raises:

    Time Complexity:
        O(n) where n is the number of elements of the data

    Space Complexity:
        O(n) where n is the number of elements of the data
    """
    return np.nanmin(data, axis)


def max(data, axis=0):
    """
    Calculates the stastical max for the inputed data

    Parameters:
        data: a numpy array containing the data to calculate the max of
        axis: the axis to operate on, defaults to 0


    Returns:
        A scalar or an array containing the resulting maxs

    Raises:

    Time Complexity:
        O(n) where n is the number of elements of the data

    Space Complexity:
        O(n) where n is the number of elements of the data
    """
    return np.nanmax(data, axis)


def histogram(data, bins=10):
    """
    Groups the data into the chosen number of bins

    Parameters:
        data: a numpy array containing the data to group
        bins: the number of bins to group the data into or an array of the bins


    Returns:
        array of the values of the histogram
        bin_edges: the edges of the bin

    Raises:

    Time Complexity:
        O(N + b), N is total number of values and b is bins.

    Space Complexity:
        O(N + b)
    """
    return np.histogram(data[~np.isnan(data)], bins)


def quantile(data, q):
    """
    Groups the data into the probility or sequence of probabilities to compute

    Parameters:
        data: a numpy array containing the data to calculate the mean of
        q: the probability or sequence of probabilites the data will be grouped into


    Returns:
        Scalar or array of the quantiles

    Raises:

    Time Complexity:
        O(N * f), or greater depending on quantile operation.

    Space Complexity:
        O(N * f)
    """
    return np.nanquantile(data, q)


class StreamingStats:
    """
    This class is responsible for incrementally calculating the
    descriptive stats across the streamed chunks of numerical
    data.

    It calculates running counts, means, variance, standard deviation,
    maximum and minimum, updating them without recalculating
    them from previous chunks. It also retains previously processed chunks
    to support the exact median, quantile and histogram calculations.

    Methods:
        update_stats(chunk_X):
        Updates the cumulative statistics using a new chunk.

        mean():
            Returns the cumulative feature means.

        variance():
            Returns the cumulative population variance.

        stan_dev():
            Returns the cumulative population standard deviation.

        min():
            Returns the cumulative feature minimums.

        max():
            Returns the cumulative feature maximums.

        median():
            Returns the median over all processed data.

        histogram():
            Returns a histogram over all processed data.

        quantile():
            Returns quantiles over all processed data.

        reset():
            Clears all accumulated statistics.
    """

    def __init__(self):
        """
        This intialises the class.

        Time Complexity:
            O(1).

        Space Complexity:
            O(1).
        """
        self.reset()

    def update_stats(self, chunk_X):
        """
        This function is responsible for updating the cumulative
        statistics using the new chunk of data.

        Any NaN values are ignored during pre-feature counts, means
        variance, minimum and maximum calculations. The chunk is also
        retained for calculating the exact median, histogram and quantile.

        Parameter:
            chunk_X:
                Is a two-dimnesional numeric arra-like that contains
                the new observations.

        Return:
            StreamingStats:
                The updated statistics object.

        Raises:
            TypeError, ValueError:
                If the input is invalid or its number of features
                differs from previously processed chunks.

        Time Complexity:
            O(n * f) wheren is number of rows and f is number of features.

        Space Complexity:
            O(n * f), due to additional persistent space as the chunk is
            retained.
        """
        chunk_X = check_array(chunk_X, allow_nan=True)

        n_features = chunk_X.shape[1]

        valid = ~np.isnan(chunk_X)

        chunk_count = np.sum(valid, axis=0)
        chunk_sum = np.nansum(chunk_X, axis=0)

        chunk_mean = np.divide(
            chunk_sum,
            chunk_count,
            out=np.zeros(n_features, dtype=float),
            where=chunk_count > 0,
        )

        chunk_M2 = np.nansum((chunk_X - chunk_mean) ** 2, axis=0)

        chunk_min = np.nanmin(chunk_X, axis=0)
        chunk_max = np.nanmax(chunk_X, axis=0)

        # The first chunk of data
        if self.count_ is None:
            self.n_features_in_ = n_features
            self.count_ = chunk_count
            self.mean_ = chunk_mean
            self.M2_ = chunk_M2
            self.min_ = chunk_min
            self.max_ = chunk_max
        else:
            if n_features != self.n_features_in_:
                raise ValueError(
                    "The number of features in chunk_X must match"
                    " the previously observed data."
                )
            old_count = self.count_
            total_count = old_count + chunk_count

            delta = chunk_mean - self.mean_

            valid_update = (old_count > 0) & (chunk_count > 0)

            self.mean_[valid_update] += (
                delta[valid_update] * chunk_count[valid_update]
            ) / total_count[valid_update]

            self.M2_[valid_update] += (
                chunk_M2[valid_update]
                + (delta[valid_update] ** 2)
                * old_count[valid_update]
                * chunk_count[valid_update]
                / total_count[valid_update]
            )
            valid_first = (old_count == 0) & (chunk_count > 0)

            self.mean_[valid_first] = chunk_mean[valid_first]
            self.M2_[valid_first] = chunk_M2[valid_first]

            self.min_ = np.fmin(self.min_, chunk_min)
            self.max_ = np.fmax(self.max_, chunk_max)

            self.count_ = total_count

        self.n_samples_seen_ += chunk_X.shape[0]
        self.chunks_data_.append(chunk_X.copy())

        return self

    def mean(self):
        """
        Calculates the cumulative mean for each feature data
        across all processed chunks

        Returns:
            ndarray:
                An array containing cumulative mean for each
                feature.

        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        return self.mean_

    def variance(self):
        """
        Calculates the cumulative population variance for each feature
        across all processed chunks.
        Returns:
            A ndarray or None, array containing the cumulative variance
            for each feature, and None if no data processed.

        Time Complexity:
            O(f), f is the number of features.

        Space Complexity:
            O(f), for the returned array.
        """

        if self.count_ is None:
            return None

        return np.divide(
            self.M2_, self.count_, out=np.zeros_like(self.M2_), where=self.count_ > 0
        )

    def stan_dev(self):
        """
        Calculates the cumulative standard deviation for each feature
        across all processed chunks.

        Returns:
            A ndarray containing the cumulative SD for each feature.

        Time Complexity:
            O(f), f is the number of features.

        Space Complexity:
            O(f), for the returned array.
        """

        if self.count_ is None:
            return None

        return np.sqrt(self.variance())

    def min(self):
        """
        Calculates the cumulative min value for each feature across
        all processed chunks.

        Returns:
            A ndarray containing the cumulative minimum value for
            each feature.

        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        return self.min_

    def max(self):
        """
        Calculates the cumulative max value for each feature across
        all processed chunks.

        Returns:
            A ndarray containing the cumulative maximum value for
            each feature.

        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        return self.max_

    def _all_data(self):
        """
        This helper function combines the previously proccessed chunks
        into one array to be used for calculating median, quantiles
        and histogram.

        Return:
            An ndarray containing all processed observations, and returns
            an empty array is no chucks have been processed.

        Time Complexity:
            O(N * f) wher N is total number of observations stored and f
            is number of features

        Space Complexity:
            O(N * f)
        """
        if len(self.chunks_data_) == 0:
            return np.array([])

        return np.concatenate(self.chunks_data_, axis=0)

    def median(self, axis=0):
        """
        Calculates the cumulative median for each feature across
        all processed chunks.

        Parameters:
            axis: the axis to operate on, defaults to 0

        Returns:
            An ndarray containing the cumulative median for each
            feature.

        Time Complexity:
            O(n) where n is the number of elements

        Space Complexity:
            O(n) where n is the number of elements
        """
        return np.nanmedian(self._all_data(), axis=axis)

    def histogram(self, bins=10):
        """
        Groups the data into the chosen number of bins

        Parameters:
            bins: the number of bins to group the data into or an array
            of the bins.

        Returns:
            array of the values of the histogram.
            bin_edges: the edges of the bin.

        Time Complexity:
            O(bn) where n is the number of elements and b is the
            number of bins.

        Space Complexity:
            O(n) where n is the number of elements.
        """
        data = self._all_data()
        return np.histogram(data[~np.isnan(data)], bins=bins)

    def quantile(self, q, axis=0):
        """
        Groups the data into the probility or sequence of probabilities to
        compute.

        Parameters:
            q: the probability or sequence of probabilites the data will
            be grouped into.

            axis: the axis to operate on, defaults to 0.

        Returns:
            A ndarray of the cumulative quantiles for each feature.

        Time Complexity:
            O(bn) where n is the number of elements of the data and b is the
            number of bins.

        Space Complexity:
            O(n) where n is the number of elements of the data.
        """
        return np.nanquantile(self._all_data(), q, axis=axis)

    def reset(self):
        """
        Clear all the statistics and stored data.

        Returns:
            None

        Time Complexity:
            O(N) where N is the stored data being released.

        Space Complexity:
            O(1)
        """
        self.n_samples_seen_ = 0
        self.n_features_in_ = None

        self.count_ = None
        self.mean_ = None
        self.M2_ = None
        self.min_ = None
        self.max_ = None

        self.chunks_data_ = []
