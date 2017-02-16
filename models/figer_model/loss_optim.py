import os
import numpy as np
import tensorflow as tf

from models.base import Model


class LossOptim(object):
  def __init__(self, figermodel):
    ''' Houses utility functions to facilitate training/pre-training'''

    # Object of the WikiELModel Class
    self.figermodel = figermodel

  #enddef init

  def make_loss_graph(self):
    self.figermodel.labeling_model.loss_graph(
      true_label_ids=self.figermodel.labels_batch,
      scope_name=self.figermodel.labeling_loss_scope,
      device_gpu=self.figermodel.device_placements['gpu'])
  #enddef decoder_losses

  def optimizer(self, optimizer_name, name):
    if optimizer_name == 'adam':
      optimizer = tf.train.AdamOptimizer(
        learning_rate=self.figermodel.learning_rate,
        name='Adam_'+name)
    elif optimizer_name == 'adagrad':
      optimizer = tf.train.AdagradOptimizer(
        learning_rate=self.figermodel.learning_rate,
        name='Adagrad_'+name)
    elif optimizer_name == 'adadelta':
      optimizer = tf.train.AdadeltaOptimizer(
        learning_rate=self.figermodel.learning_rate,
        name='Adadelta_'+name)
    elif optimizer_name == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=self.figermodel.learning_rate,
        name='SGD_'+name)
    elif optimizer_name == 'momentum':
      optimizer = tf.train.MomentumOptimizer(
        learning_rate=self.figermodel.learning_rate,
        momentum=0.9,
        name='Momentum_'+name)
    else:
      print("OPTIMIZER WRONG. HOW DID YOU GET HERE!!")
      sys.exit(0)
    return optimizer

  def weight_regularization(self, trainable_vars):
    vars_to_regularize = []
    regularization_loss = 0
    for var in trainable_vars:
      if "_weights" in var.name:
        regularization_loss += tf.nn.l2_loss(var)
        vars_to_regularize.append(var)
    #endffor

    print("L2 - Regularization for Variables:")
    self.figermodel.print_variables_in_collection(vars_to_regularize)
    return regularization_loss

  def label_optimization(self, trainable_vars, optim_scope):
    # POSTERIOR ENCODER LOSS - Only for pretraining
    self.total_loss = self.figermodel.labeling_model.labeling_loss
    self.labeling_loss = self.figermodel.labeling_model.labeling_loss

    # Weight Regularization
    self.regularization_loss = self.weight_regularization(trainable_vars)
    self.total_loss += self.figermodel.reg_constant*self.regularization_loss

    # Loss after regularization
    _ = tf.scalar_summary("loss_regularized", self.total_loss)

    # SCALAR SUMMARIES - ENCODER
    _ = tf.scalar_summary("loss_labeling", self.labeling_loss)

    with tf.variable_scope(optim_scope) as s, tf.device(self.figermodel.device_placements['gpu']) as d:
        self.optimizer = self.optimizer(
          optimizer_name=self.figermodel.optimizer, name="opt")
        self.gvs = self.optimizer.compute_gradients(
          loss=self.total_loss, var_list=trainable_vars)
        #self.clipped_gvs = self.clip_gradients(self.gvs)
        self.optim_op = self.optimizer.apply_gradients(self.gvs)
    #end variable-scopes
  #enddef pretraining_optim

  def clip_gradients(self, gvs):
    clipped_gvs = []
    for (g,v) in gvs:
      if self.figermodel.embeddings_scope in v.name:
        clipped_gvalues = tf.clip_by_norm(g.values, 30)
        clipped_index_slices = tf.IndexedSlices(
          values=clipped_gvalues,
          indices=g.indices)
        clipped_gvs.append((clipped_index_slices, v))
      else:
        clipped_gvs.append((tf.clip_by_norm(g, 1), v))
    return clipped_gvs
  #end_clipping