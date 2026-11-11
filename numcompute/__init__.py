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
from .metrics import (
    AUC,
    F1,
    MSE,
    Accuracy,
    ConfusionMatrix,
    Precision,
    Recall,
    RollingWindowMetric,
)
from .optim import grad, jacobian
from .pipeline import Pipeline
from .preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
from .rank import (
    percentile,
    rank,
)
from .sort_search import (
    binary_search,
    multi_key_sort,
    quickselect,
    stable_sort,
    topk,
)
from .stats import (
    StreamingStats,
    histogram,
    max,
    mean,
    median,
    min,
    quantile,
    stan_dev,
)
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier
from .visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)

__all__ = [
    "load_csv",
    "grad",
    "jacobian",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",
    "StandardScaler",
    "stable_sort",
    "multi_key_sort",
    "topk",
    "quickselect",
    "binary_search",
    "rank",
    "percentile",
    "mean",
    "median",
    "stan_dev",
    "min",
    "max",
    "quantile",
    "histogram",
    "StreamingStats",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ConfusionMatrix",
    "MSE",
    "AUC",
    "RollingWindowMetric",
    "DecisionTreeClassifier",
    "Pipeline",
    "StreamTrainer",
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
]
