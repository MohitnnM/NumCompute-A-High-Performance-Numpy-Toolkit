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
def mean(data):
    return np.mean(data)

def median(data):
    return np.median(data)

def stan_dev(data):
    return np.std(data)

def min(data):
    return np.min(data)

def max(data):
    return np.max(data)
