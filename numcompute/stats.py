"""
Provides descriptive statistics for the numerical arrays for analysis 
components.

It calculates basic statistics such as mean, median, standard deviation,
min and max. It also groups observations into bins, counts how many 
observations fall into each interval and also uses quantiles to identify
values that divide ordered observations into specified proportions. 

The basic statistics must follow axis-wise statistics with clear dimension
and shape behaviour, and the quantiles must also handle NaN values 
appropriately.  
"""
import numpy as np

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
    return np.mean(data,axis)

def median(data,axis):
    """
        Calculates the stastical median for the inputed data

        Parameters:
            data: a numpy array containing the data to calculate the median of
            axis: the axis to operate on, defaults to 0
            

        Returns:
            A scalar or an array containing the resulting median

        Raises:
        
        Time Complexity:
            O(n) where n is the number of elements of the data

        Space Complexity:
            O(n) where n is the number of elements of the data
    """  
    return np.median(data,axis)

def stan_dev(data,axis):
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
    return np.std(data,axis)

def min(data, axis):
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
    return np.min(data,axis)

def max(data, axis):
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
    return np.max(data, axis)

def histogram(data, bins=10):
    """
        Groups the data into the chosen number of bins

        Parameters:
            data: a numpy array containing the data to calculate the mean of
            axis: the axis to operate on, defaults to 0
            

        Returns:
            array of the values of the histogram
            bin_edges: the edges of the bin

        Raises:
        
        Time Complexity:
            O(bn) where n is the number of elements of the data and b is the number of bins

        Space Complexity:
            O(n) where n is the number of elements of the data
    """
    return np.histogram(data,bins)

def quantile(data, q):
    """
        Groups the data into the probility or sequence of probabilities to compute

        Parameters:
            data: a numpy array containing the data to calculate the mean of
            axis: the axis to operate on, defaults to 0
            

        Returns:
            Scalar or array of the quantiles

        Raises:
        
        Time Complexity:
            O(bn) where n is the number of elements of the data and b is the number of bins

        Space Complexity:
            O(n) where n is the number of elements of the data
    """
    return np.quantile(data, q)