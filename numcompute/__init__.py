"""
NumCompute: A scientific computing toolkit built with NumPy.

NumCompute delivers standardised modules for things like data reading and
processing, sorting and searching, ranking, descriptive statistics, metrics
for evaluation, gradient estimation, a pipeline for transformation and
benchmarking.

The purpose of this package is to simulate core scientific computing and
machine learning workflows using only Python and NumPy, focusing on
vectorisation, numerical stability, consistent APIs, and reproducible behaviour.
"""

from .io import load_csv
from .optim import grad, jacobian
from .preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .stats import mean, median, stan_dev, min, max, quantile, histogram
from .metrics import accuracy, precision, recall, f1score, confusion_matrix, MSE

__all__ = [
    "load_csv",
    "grad",
    "jacobian",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",
    "StandardScaler",
    "mean",
    "median",
    "stan_dev",
    "min",
    "max",
    "quantile",
    "histogram"
]
    "accuracy",
    "precision",
    "recall",
    "f1score",
    "confusion_matrix",
    "MSE"
]
