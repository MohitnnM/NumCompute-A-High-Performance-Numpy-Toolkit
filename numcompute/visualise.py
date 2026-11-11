"""
This module provides reusable Matplotlib visualisation functions
for streaming machine-learning results.

It supports plotting metrics across chunks of data, comparing the
performance of streaming models and also visualising the predictions
against ground-truth labels.

The option to save plots locally is available.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_over_time(metric_values, title, ylabel, save_path=None, show=True):
    """
    This function plots the metric values across sequenctial streaming
    chunks of data.

    Parameters:
        metric_values:
            Is the sequence of metric values to plot.

        title:
            Is the title of the plot.

        ylabel:
            Is the y axis label.

        save_path:
            Is the path of where to save the file locally, set to
            None as default.

        show:
            Boolean to decide if the plot should be shown.

        Returns:
            A tuple that contains the Matplotlip figure and axis objects.

    Time Complexity:
        O(n), n is the number of plotted observations.

    Space Complexity:
        O(n), for plot data and Matplotlib objects
    """

    metric_values = np.asarray(metric_values, dtype=float)

    chunks = np.arange(1, metric_values.size + 1)

    figure, axis = plt.subplots()

    axis.plot(chunks, metric_values, marker="o")

    axis.set_title(title)
    axis.set_xlabel("Chunk")
    axis.set_ylabel(ylabel)
    axis.grid(True)

    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return figure, axis


def compare_models(metric1, metric2, labels, save_path=None, show=True):
    """
    This function plots the streaming metric values of two models
    across chunks of data.

    Parameters:
        metric1:
            Is the sequence of metric values for the first model.

        metric2:
            Is the sequence of metric values for the second model.

        label:
            Is the labels for the two models axis label.

        save_path:
            Is the path of where to save the file locally, set to
            None as default.

        show:
            Boolean to decide if the plot should be shown.

    Returns:
        A tuple that contains the Matplotlip figure and axis objects.

    Raises:
        ValueError:
            If the metric sequence has different lengths or the label
            does not contain exatly 2 values.

    Time Complexity:
        O(n), n is the number of plotted observations.

    Space Complexity:
        O(n), for plot data and Matplotlib objects
    """

    metric1 = np.asarray(metric1, dtype=float)
    metric2 = np.asarray(metric2, dtype=float)

    if metric1.size != metric2.size:
        raise ValueError(
            "Both metric1 and metric2 must have the same number of values."
        )

    if len(labels) != 2:
        raise ValueError("Labels must have 2 values.")

    chunks = np.arange(1, metric1.size + 1)

    figure, axis = plt.subplots()

    axis.plot(chunks, metric1, marker="o", label=labels[0])
    axis.plot(chunks, metric2, marker="o", label=labels[1])

    axis.set_title("Compare Models")
    axis.set_xlabel("Chunk")
    axis.set_ylabel("Metric")

    axis.legend()
    axis.grid(True)

    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return figure, axis


def plot_predictions_vs_ground_truth(y_true, y_pred, save_path=None, show=True):
    """
    This function plots the predicted class labels against the
    ground-truth class lables.

    Parameters:
        y_true:
            Is the sequence containing the true class labels.

        y_pred:
            Is the sequence containing the prediction class labels.

        save_path:
            Is the path of where to save the file locally, set to
            None as default.

        show:
            Boolean to decide if the plot should be shown.

    Returns:
        A tuple that contains the Matplotlip figure and axis objects.

    Raises:
        ValueError:
            If y_true and y_pred have different shapes.

    Time Complexity:
        O(n), n is the number of plotted observations.

    Space Complexity:
        O(n), for plot data and Matplotlib objects
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    if y_pred.shape != y_true.shape:
        raise ValueError("Both y_pred and y_true must have the same shape.")

    observations = np.arange(y_true.size)

    figure, axis = plt.subplots()

    axis.plot(observations, y_true, marker="o", label="Ground Truth")
    axis.plot(observations, y_pred, marker="o", label="Predictions")

    axis.set_title("Predictions vs Ground Truth")
    axis.set_xlabel("Observation")
    axis.set_ylabel("Class")

    axis.legend()
    axis.grid(True)

    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return figure, axis
