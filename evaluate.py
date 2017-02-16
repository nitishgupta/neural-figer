import os
import sys
import numpy as np

def true_and_prediction(true_label_batch, pred_score_batch):
  ''' Gets true labels and pred scores in numpy matrix and converts to list
  args
    true_label_batch: Binary Numpy matrix of [num_instances, num_labels]
    pred_score_batch: Real [0,1] numpy matrix of [num_instances, num_labels]

  return:
    true_labels: List of list of true label (indices) for batch of instances
    pred_labels : List of list of pred label (indices) for batch of instances
      (threshold = 0.5)
  '''
  truebool = true_label_batch == 1.0
  predbool = pred_score_batch >= 0.5
  truebool = truebool.tolist()
  predbool = predbool.tolist()
  assert len(truebool) == len(predbool), "Num of instances dont match"
  num_instanes = len(truebool)
  true_labels = []
  pred_labels = []
  for i in range(0, num_instanes):
    true_labels_i = [i for i, x in enumerate(truebool[i]) if x]
    true_labels.append(true_labels_i)
    pred_labels_i = [i for i, x in enumerate(predbool[i]) if x]
    pred_labels.append(pred_labels_i)
  ##
  return (true_labels, pred_labels)


def strict_pred(true_label_batch, pred_score_batch):
  ''' Calculates strict precision/recall/f1 given truth and predicted scores
  args
    true_label_batch: Binary Numpy matrix of [num_instances, num_labels]
    pred_score_batch: Real [0,1] numpy matrix of [num_instances, num_labels]

  return:
    correct_preds: Number of correct strict preds
    precision : correct_preds / num_instances
  '''
  (true_labels, pred_labels) = true_and_prediction(
    true_label_batch, pred_score_batch)

  num_instanes = len(true_labels)
  correct_preds = 0
  for i in range(0, num_instanes):
    if true_labels[i] == pred_labels[i]:
      correct_preds += 1
  #endfor
  precision = recall = float(correct_preds)/num_instanes

  return correct_preds, precision






