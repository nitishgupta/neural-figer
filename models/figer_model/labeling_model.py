import time
import tensorflow as tf
import numpy as np

from models.base import Model

class LabelingModel(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, batch_size, num_labels, context_encoded_dim,
               context_encoded, scope_name, device):

    self.batch_size = batch_size
    self.num_labels = num_labels
    with tf.variable_scope(scope_name) as s, tf.device(device) as d:
      self.label_weights = tf.get_variable(
        name="label_weights",
        shape=[context_encoded_dim, num_labels],
        initializer=tf.random_normal_initializer(mean=0.0,
                                                 stddev=1.0/(100.0)))

      # [B, L]
      self.label_scores = tf.matmul(context_encoded, self.label_weights)
      self.label_probs = tf.sigmoid(self.label_scores)


  def loss_graph(self, true_label_ids, scope_name, device_gpu):
    with tf.variable_scope(scope_name) as s, tf.device(device_gpu) as d:
      # [B, L]
      self.cross_entropy_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.label_scores,
        targets=true_label_ids,
        name="labeling_loss")

      self.labeling_loss = tf.reduce_sum(
        self.cross_entropy_losses) / tf.to_float(self.batch_size)*tf.to_float(self.num_labels)