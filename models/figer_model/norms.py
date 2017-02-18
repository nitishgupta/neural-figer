import os
import numpy as np
import tensorflow as tf

from models.base import Model

class Norms(Model):
  def __init__(self, figermodel):
    context_encoded_dim = figermodel.context_encoded_dim
    lstm_size = figermodel.context_encoder_lstmsize
    bs = figermodel.batch_size
    num_labels = figermodel.num_labels

    self.left_lstm_norm = self.norm(
      figermodel.context_encoder_model.left_last_output)/(bs*lstm_size)
    self.right_lstm_norm = self.norm(
      figermodel.context_encoder_model.right_last_output)/(bs*lstm_size)

    self.context_encoded_norm = self.norm(
      figermodel.context_encoder_model.context_encoded)/(bs*context_encoded_dim)

    self.label_scores_norm = self.norm(
      figermodel.labeling_model.label_scores)/(bs*num_labels)

    self.norms = [self.left_lstm_norm, self.right_lstm_norm,
                  self.context_encoded_norm, self.label_scores_norm]

  def norm(self, var):
    norm = tf.sqrt(tf.reduce_sum(tf.pow(var, 2)))
    return norm




