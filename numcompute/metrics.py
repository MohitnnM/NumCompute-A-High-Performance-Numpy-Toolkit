"""
Evaluates the predicted values against known targets using classification and 
regression metrics.

For classification, this module will evaluate the predicted values using 
performance metrics like:

  - Accuracy, which calculates the proportion of predictions that are correct.

  - Precision, which measures the proportion of predicted positive cases that 
    were actually positive.

  - Recall, which measures the proportion of actual positive cases that the 
    model correctly identified.

  - F1 score, which combines precision and recall to evaluate the model on 
    a range of [0,1].

  - Confusion matrix, which summarises the number of correct and incorrect 
    predictions for each class. 

For regression, this module evaluates the numerical predictions using:
  - The Mean Squared Error (MSE), which evaluates the model by calculating
    the average squared difference between the predicted values and the target
    values. 

Metrics in this module are designed to operate on NumPy arrays and provide 
consistent behaviour during comparison of predicted values and their related
target values. 
"""
