import os
import sys
import numpy as np


''' Location : 1, Organization : 5, Person : 9, Event : 29] '''
coarseTypeIds = [1, 5, 9, 29]

def types_convert_mat_to_sets(true_label_batch, pred_score_batch):
  ''' Gets true labels and pred scores in numpy matrix and converts to list
  args
    true_label_batch: Binary Numpy matrix of [num_instances, num_labels]
    pred_score_batch: Real [0,1] numpy matrix of [num_instances, num_labels]

  return:
    true_labels: List of list of true label idxs for batch of instances
    pred_labels : List of list of pred label idxs for batch of instances
      using (threshold = 0.5)
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
    true_labels.append(set(true_labels_i))
    pred_labels_i = [i for i, x in enumerate(predbool[i]) if x]
    pred_labels.append(set(pred_labels_i))
  ##
  return (true_labels, pred_labels)

def types_prediction_stats(true_labels, pred_labels):
  '''
  args
    true_label_batch: Binary Numpy matrix of [num_instances, num_labels]
    pred_score_batch: Real [0,1] numpy matrix of [num_instances, num_labels]
  '''
  #(true_labels, pred_labels) = true_and_prediction(true_label_batch,
  #                                                 pred_score_batch)

  # t_hat \interesect t
  t_intersect = 0
  t_hat_count = 0
  t_count = 0
  t_t_hat_exact = 0
  loose_macro_p = 0.0
  loose_macro_r = 0.0
  assert len(true_labels) == len(pred_labels)
  num_instances = len(true_labels)
  for i in range(0, num_instances):
    intersect = len(true_labels[i].intersection(pred_labels[i]))
    t_h_c = len(pred_labels[i])
    t_c = len(true_labels[i])
    t_intersect += intersect
    t_hat_count += t_h_c
    t_count += t_c
    exact = 1 if (true_labels[i] == pred_labels[i]) else 0
    t_t_hat_exact += exact
    if len(pred_labels[i]) > 0:
      loose_macro_p += intersect / float(t_h_c)
    if len(true_labels[i]) > 0:
      loose_macro_r += intersect / float(t_c)

  return t_intersect, t_t_hat_exact, t_hat_count, t_count, loose_macro_p, loose_macro_r


def types_predictions(true_label_batches, pred_score_batches):
  ''' Get lists of batches for true_labels and pred_scores
  Each element in list is a numpy matrix
    true_label_batches[i] : Binary Numpy matrix of [num_instances, num_labels]
    pred_score_batches[i] : Real [0,1] numpy matrix of [num_instances, num_labels]
  '''
  assert len(true_label_batches) == len(pred_score_batches)
  num_instances = 0
  # (true_label, pred_labels) = list of set of (true_label_idxs, pred_label_idxs)
  (true_labels, pred_labels) = ([], [])
  for i in range(0, len(true_label_batches)):
    # Break batch into list of true labels and pred labels for each sample
    (true_labels_bi, pred_labels_bi) = types_convert_mat_to_sets(
      true_label_batches[i], pred_score_batches[i])
    true_labels.extend(true_labels_bi)
    pred_labels.extend(pred_labels_bi)
    num_instances += len(true_labels_bi)
  #
  (t_i, t_th_exact, t_h_c, t_c, l_m_p, l_m_r) = types_prediction_stats(
    true_labels, pred_labels)

  strict = float(t_th_exact)/float(num_instances)
  loose_macro_p = l_m_p / float(num_instances)
  loose_macro_r = l_m_r / float(num_instances)
  loose_macro_f = f1(loose_macro_p, loose_macro_r)
  loose_micro_p = float(t_i)/float(t_h_c)
  loose_micro_r = float(t_i)/float(t_c)
  loose_micro_f = f1(loose_micro_p, loose_micro_r)

  print("Strict : {}".format(strict))
  print("Loose Macro P : {0:.3f}  R : {1:.3f}  F : {2:.3f}".format(loose_macro_p, loose_macro_r, loose_macro_f))
  print("Loose Micro P : {0:.3f}  R : {1:.3f}  F : {2:.3f}".format(loose_micro_p, loose_micro_r, loose_micro_f))



def f1(p,r):
  if p == 0.0 and r == 0.0:
    return 0.0
  return (float(2*p*r))/(p + r)

def strict_pred(true_label_batch, pred_score_batch):
  ''' Calculates strict precision/recall/f1 given truth and predicted scores
  args
    true_label_batch: Binary Numpy matrix of [num_instances, num_labels]
    pred_score_batch: Real [0,1] numpy matrix of [num_instances, num_labels]

  return:
    correct_preds: Number of correct strict preds
    precision : correct_preds / num_instances
  '''
  (true_labels, pred_labels) = types_convert_mat_to_sets(
    true_label_batch, pred_score_batch)

  num_instanes = len(true_labels)
  correct_preds = 0
  for i in range(0, num_instanes):
    if true_labels[i] == pred_labels[i]:
      correct_preds += 1
  #endfor
  precision = recall = float(correct_preds)/num_instanes

  return correct_preds, precision






