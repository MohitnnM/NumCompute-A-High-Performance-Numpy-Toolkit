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
import numpy as np

def accuracy(y_pred, y_true):
    return np.divide(np.sum(np.where(y_pred ==y_true,1,0)),y_true.size)

def precision(y_pred, y_true, pred_class):
  pos_index = y_pred == pred_class
  return np.divide(np.sum(y_pred[pos_index] == y_true[pos_index]), y_pred[pos_index].size)

def recall(y_pred, y_true, pred_class):
  pos_index = y_true == pred_class
  return np.divide(np.sum(y_pred[pos_index] == y_true[pos_index]), y_true[pos_index].size)

def f1score(y_pred, y_true, pred_class):
   prec = precision(y_pred,y_true,pred_class)
   reca = recall(y_pred,y_true,pred_class)
   return 2* ((prec*reca)/(prec+reca))

def confusion_mtraix(y_pred, y_true, pred_class):
   pred_pos = y_pred == pred_class
   pred_neg = y_pred != pred_class
   true_pos = np.count_nonzero(y_pred[pred_pos] == y_true[pred_pos])
   true_neg = np.count_nonzero(y_pred[pred_neg] == y_true[pred_neg])
   false_pos = np.count_nonzero(y_pred[pred_pos] != y_true[pred_pos])
   flase_neg = np.count_nonzero(y_pred[pred_neg] != y_true[pred_neg])
   return (true_pos,false_pos,true_neg,flase_neg)

def MSE(y_pred, y_true):
   return np.divide(np.sum(np.power(y_pred-y_true,2)),y_true.size)
