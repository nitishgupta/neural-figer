import time
import tensorflow as tf
import numpy as np
import random
import sys
import gc

import evaluate
from models.base import Model
from models.figer_model.context_encoder import ContextEncoderModel
from models.figer_model.labeling_model import LabelingModel
from models.figer_model.loss_optim import LossOptim


np.set_printoptions(precision=5)


class FigerModel(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, sess, reader, dataset, max_steps, pretrain_max_steps,
               word_embed_dim, context_encoded_dim,
               context_encoder_lstmsize, context_encoder_num_layers,
               learning_rate, dropout_keep_prob, reg_constant, checkpoint_dir,
               optimizer, mode='train'):
    self.optimizer = optimizer
    self.mode = mode
    self.sess = sess
    self.reader = reader  # Reader class
    self.dataset = dataset

    self.max_steps = max_steps  # Max num of steps of training to run
    self.pretrain_max_steps = pretrain_max_steps
    self.batch_size = reader.batch_size
    self.reg_constant = reg_constant
    self.dropout_keep_prob = dropout_keep_prob

    # Num of clusters = Number of entities in dataset.
    self.num_labels = self.reader.num_labels
    self.num_words = self.reader.num_words

    # Size of word embeddings
    self.word_embed_dim = word_embed_dim

    # Context encoders
    self.context_encoded_dim = context_encoded_dim
    self.context_encoder_lstmsize = context_encoder_lstmsize
    self.context_encoder_num_layers = context_encoder_num_layers

    self.checkpoint_dir = checkpoint_dir

    self.embeddings_scope = "embeddings"
    self.word_embed_var_name = "word_embeddings"
    self.encoder_model_scope = "context_encoder"
    self.label_model_scope = "labeling_model"
    self.labeling_loss_scope = "labeling_loss"
    self.optim_scope = "labeling_optimization"

    self._attrs=[
      "word_embed_dim", "num_words", "num_labels",
      "context_encoded_dim", "context_encoder_lstmsize",
      "context_encoder_num_layers", "reg_constant", "optimizer"]

    #GPU Allocations
    self.device_placements = {
      'cpu': '/cpu:0',
      'gpu': '/gpu:0'
    }

    with tf.variable_scope("figer_model") as scope:
      self.learning_rate = tf.Variable(learning_rate, name='learning_rate',
                                       trainable=False)
      self.global_step = tf.Variable(0, name='global_step', trainable=False,
                                     dtype=tf.int32)
      self.increment_global_step_op = tf.assign(self.global_step, self.global_step+1)


      self.build_placeholders()

      # Encoder Models : Name LSTM, Text FF and Links FF networks
      with tf.variable_scope(self.encoder_model_scope) as scope:
        self.context_encoder_model = ContextEncoderModel(
          num_layers=self.context_encoder_num_layers,
          batch_size=self.batch_size,
          word_embeddings=self.word_embeddings,
          lstm_size=self.context_encoder_lstmsize,
          left_batch=self.left_batch,
          left_lengths=self.left_lengths,
          right_batch=self.right_batch,
          right_lengths=self.right_lengths,
          context_encoded_dim=self.context_encoded_dim,
          scope_name=self.encoder_model_scope,
          device=self.device_placements['gpu'],
          dropout_keep_prob=self.dropout_keep_prob)

        self.labeling_model = LabelingModel(
          batch_size=self.batch_size,
          num_labels=self.num_labels,
          context_encoded_dim=self.context_encoded_dim,
          context_encoded=self.context_encoder_model.context_encoded,
          scope_name=self.label_model_scope,
          device=self.device_placements['gpu'])
      #end - encoder variable scope


    # Encoder FF Variables + Cluster Embedding
    self.train_vars = tf.trainable_variables()

    self.vars_to_store = []
    self.vars_to_store.extend(
      self.scope_vars_list(scope_name="figer_model",
                           var_list=tf.all_variables()))

    print("All Trainable Variables")
    self.print_variables_in_collection(
      self.train_vars)

    print("All Variables getting stored")
    self.print_variables_in_collection(
      self.vars_to_store)

    self.loss_optim = LossOptim(self)
  ################ end Initialize  #############################################


  def build_placeholders(self):
    self.left_batch = tf.placeholder(tf.int32,
                                     [self.batch_size, None],
                                     name="left_batch")
    self.left_lengths = tf.placeholder(tf.int32,
                                       [self.batch_size],
                                       name="left_lengths")
    self.right_batch = tf.placeholder(tf.int32,
                                      [self.batch_size, None],
                                      name="right_batch")
    self.right_lengths = tf.placeholder(tf.int32,
                                        [self.batch_size],
                                        name="right_lengths")

    self.labels_batch = tf.placeholder(tf.float32,
                                       [self.batch_size, self.num_labels],
                                       name="true_labels")
    #END-Placeholders

    with tf.variable_scope(self.embeddings_scope) as s:
      with tf.device(self.device_placements['gpu']) as d:
        self.word_embeddings = tf.get_variable(
          name=self.word_embed_var_name,
          shape=[self.num_words, self.word_embed_dim],
          initializer=tf.random_normal_initializer(
            mean=0.0, stddev=(1.0/100.0)))

  def training_setup(self):
    # Make the loss graph
    print("[#] Making Loss Graph ....")
    self.loss_optim.make_loss_graph()

    print("[#] Defining pretraining losses and optimizers ...")
    self.loss_optim.label_optimization(
      trainable_vars=self.train_vars,
      optim_scope=self.optim_scope)
    #self.norms = Norms(self)

    print("[#] Initializing pretraining optimizers variables ...")
    self.optim_vars = self.scope_vars_list(scope_name=self.optim_scope,
                                           var_list=tf.all_variables())

    # All Variables - Vars_to_Store + Optim Variables
    variables_to_store = self.optim_vars + self.vars_to_store
    print("Variables being stored (after optimizers) ... ")
    self.print_variables_in_collection(variables_to_store)

    return variables_to_store

  def training(self):
    vars_tostore = self.training_setup()

    saver = tf.train.Saver(var_list=vars_tostore, max_to_keep=5)

    # (Try) Load all pretraining model variables
    # If pre-training graph not found - Initialize trainable + optim variables
    print("Loading pre-saved checkpoint...")
    load_status = self.load_wsaver(saver=saver,
                                   checkpoint_dir=self.checkpoint_dir,
                                   attrs=self._attrs)
    if not load_status:
      self.sess.run(tf.initialize_variables(vars_tostore))

    start_iter = self.global_step.eval()
    start_time = time.time()
    merged_sum = tf.merge_all_summaries()
    log_dir = self.get_log_dir(root_log_dir="./logs/")
    writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

    print("[#] Pre-Training iterations done: %d" % start_iter)

    data_loading = 0

    tf.get_default_graph().finalize()

    for iteration in range(start_iter, self.max_steps):
      dstime = time.time()
      (left_batch, left_lengths,
       right_batch, right_lengths, labels_batch) = self.reader.next_train_batch()

      dtime = time.time() - dstime
      data_loading += dtime

      feed_dict = {self.left_batch: left_batch,
                   self.left_lengths: left_lengths,
                   self.right_batch: right_batch,
                   self.right_lengths: right_lengths,
                   self.labels_batch: labels_batch}

      fetch_tensors = [self.loss_optim.labeling_loss,
                       self.labeling_model.label_probs]
      '''
      norm_tensors = [self.norms.text_norm,
                      self.norms.doc_norm,
                      self.norms.links_norm,
                      self.norms.context_norm,
                      self.norms.true_embed_norm,
                      self.norms.neg_embed_norm,
                      self.norms.en_ff_varnorms,
                      self.norms.en_ff_gradnorms]
      fetch_tensors.append(norm_tensors)
      '''

      (fetches_old,
       _,
       _,
       summary_str) = self.sess.run([fetch_tensors,
                                     self.loss_optim.optim_op,
                                     self.increment_global_step_op,
                                     merged_sum],
                                    feed_dict=feed_dict)

      fetches_new = self.sess.run(fetch_tensors,
                                  feed_dict=feed_dict)

      [old_loss, old_label_sigms] = fetches_old
      [new_loss, new_label_sigms] = fetches_new
      '''
      [text_norm,
       doc_norm,
       links_norm,
       context_norm,
       true_embed_norm,
       neg_embed_norm,
       en_ff_varnorms,
       en_ff_gradnorms] = fetches[5]
      en_ff_varnames = self.norms.en_ff_varnames
      if self.decoder_bool:
        decoder_pretraining_loss = fetches[6]
      '''

      #self.global_step.assign(iteration).eval()

      if iteration % 100 == 0:
        # [B, L]
        old_corr_preds, old_precision = evaluate.strict_pred(
          labels_batch, old_label_sigms)
        new_corr_preds, new_precision = evaluate.strict_pred(
          labels_batch, new_label_sigms)

        print("Iter %2d, Epoch %d, T %4.2f secs, Loss %.3f, New_Loss %.3f"
              % (iteration, self.reader.tr_epochs, time.time() - start_time,
                 old_loss, new_loss))
        print("[OLD] Num of strict correct predictions : {}, {}".format(
          old_corr_preds, old_precision))
        # print("[NEW] Num of strict correct predictions : {}, {}".format(
        #   old_corr_preds, old_precision))
        print("Time to load data : %4.2f\n" % data_loading)
        data_loading = 0
      #end-100

      if iteration % 2 == 0:
       writer.add_summary(summary_str, iteration)

      if iteration != 0 and iteration % 2000 == 0:
        self.save_wsaver(saver=saver, checkpoint_dir=self.checkpoint_dir,
                         attrs=self._attrs,
                         global_step=self.global_step)
        self.validation()
  #end pretraining

  def validation(self):
    print("Validation accuracy starting ... ")
    self.reader.reset_validation()
    total_correct_preds = 0
    total_instances = 0

    stime = time.time()
    while self.reader.val_epochs < 1:
      (left_batch, left_lengths,
       right_batch, right_lengths, labels_batch) = self.reader.next_val_batch()

      feed_dict = {self.left_batch: left_batch,
                   self.left_lengths: left_lengths,
                   self.right_batch: right_batch,
                   self.right_lengths: right_lengths,
                   self.labels_batch: labels_batch}

      fetch_tensors = [self.loss_optim.labeling_loss,
                       self.labeling_model.label_probs]

      fetches = self.sess.run(fetch_tensors,
                                  feed_dict=feed_dict)


      [loss, label_sigms] = fetches

      correct_preds, precision = evaluate.strict_pred(labels_batch, label_sigms)
      total_correct_preds += correct_preds
      total_instances += self.reader.batch_size
    #endwhile
    ttime = float(time.time() - stime)/60.0
    precision = float(total_correct_preds)/float(total_instances)
    print("Num of instances {0} Strict Precision : {1:.4f} T {2:.3f} mins".format(
      total_instances, precision, ttime))
  #end validation


  ######################      TESTING     #####################################
  def testing(self):
    assert self.batch_size == 1, "Some code in testing hard coded for B=1"
    # Make the loss graph
    if self.decoder_bool:
      print("[#] Making Decoder Loss Graphs ....")
      self.loss_optim.make_decoder_loss_graphs()

    print("[#] Making Pretraining encoder Loss Graph ....")
    self.loss_optim.make_pretraining_encoder_loss_graph()


    # (Try) Load the pretraining model trainable variables
    # If pre-training graph not found - Initialize trainable variables
    print("Loading pre-training checkpoint...")
    load_status = self.load(checkpoint_dir=self.checkpoint_dir,
                            var_list=self.pretraining_vars,
                            attrs=self._pretrain_attrs)
    if not load_status:
      print("PRETRAINED CHECKPOINT NOT FOUND. QUITTING ... ")
      sys.exit(0)


    pretrained_iter = self.pretrain_global_step.eval()
    print("[#] Pre-Training iterations done: %d" % pretrained_iter)
    start_time = time.time()
    iteration = 0

    total_preds = 0
    correct_crosswikis = 0
    correct_wo_cprob = 0
    correct_w_cprob = 0
    count_not_in_crosswikis = 0

    data_loading = 0
    while self.reader.epochs < 1:
      iteration += 1
      dstime = time.time()

      (name_batch, dec_in_name_batch,
       mids_batch, wids_batch, wid_cprobs_batch,
       cands_batch, cands_cprobs_batch,
       text_batch, doc_batch, links_batch,
       docs_out_batch, links_out_batch) = self.reader.next_test_batch()

      #neg_samples = self.make_neg_samples(mids_batch)
      dtime = time.time() - dstime
      data_loading += dtime

      feed_dict = {self.text_batch: text_batch,
                   self.doc_batch: doc_batch,
                   self.links_batch: links_batch,
                   self.mids_batch: wids_batch,
                   self.sampled_negative_cluster_ids: cands_batch}

      fetch_tensors = [self.posterior_model.posterior_loss,
                       self.posterior_model.true_cluster_scores,
                       self.posterior_model.true_cluster_sigmoids,
                       self.posterior_model.neg_clusters_scores,
                       self.posterior_model.neg_clusters_sigmoids,
                       self.posterior_model.all_scores]

      [fetches] = self.sess.run([fetch_tensors],
                                feed_dict=feed_dict)

      encoder_pretraining_loss = fetches[0]
      true_scores = fetches[1]
      true_sigmoids = fetches[2]
      neg_scores = fetches[3]
      neg_sigmoids = fetches[4]
      all_scores = fetches[5]

      print("Iteration: [%2d] Epoch: [%4d] time: %4.2f, "
            "encoder pretrain loss: %.5f, "
            % (iteration, self.reader.epochs,
               time.time() - start_time,
               encoder_pretraining_loss))
      print("Time to load data : %4.2f" % data_loading)
      data_loading = 0


      # ASSERT B = 1
      true_score = true_scores[0] # Scalar
      neg_score = neg_scores[0]   # [C]
      true_wid_cprob = np.array([wid_cprobs_batch[0]])  # Scalar
      cand_wid_cprobs = np.array(cands_cprobs_batch[0])  # [C]
      context_scores = np.array(all_scores[0])  # [C+1]

      com_cprob_scores = np.concatenate((true_wid_cprob, cand_wid_cprobs)) # [C+1]


      # Softmax of Context Scores
      softmax_context = self.softmax(context_scores)
      print("Context Softmax Prob")
      print(softmax_context)
      idx_predwid = np.argmax(softmax_context, axis=0)
      max_context_score = np.amax(softmax_context, axis=0)
      if idx_predwid == 0:
        pred_wid = self.reader.idx2wid[wids_batch[0]]
      else:
        pred_wid = self.reader.idx2wid[cands_batch[0][idx_predwid-1]]
      pred_WikiTitle = self.reader.wid2wikiTitle[pred_wid]
      print("Pred WikiTitle: {} Score : {}".format(pred_WikiTitle, max_context_score))

      # Softmax(context_scores)*cprobs
      expc = np.exp(context_scores)
      expc_weighed = expc*com_cprob_scores
      sumc = np.sum(expc_weighed)
      scores_with_cprob = expc_weighed/sumc

      print("Context Scores w/ cprob")
      print(scores_with_cprob)

      total_preds += 1

      # If not found in CrossWikis
      if np.all(com_cprob_scores == 0.0):
        count_not_in_crosswikis += 1
        print("Softmax Context")
        print(softmax_context)

        print("WID|Candidate Cprobs")
        print(com_cprob_scores)

        print("WID|Candidate Cprob*Score")
        print(scores_with_cprob)
      else:
        if np.equal(np.argmax(com_cprob_scores, axis=0), 0):
          correct_crosswikis += 1

        # Without crpob
        bool_correct = np.equal(np.argmax(softmax_context, axis=0), 0)
        if bool_correct == True:
          correct_wo_cprob += 1

        # With crpob
        bool_correct = np.equal(np.argmax(scores_with_cprob, axis=0), 0)
        if bool_correct == True:
          correct_w_cprob += 1

      # Finding the true/Max scoring WikiTitle
      true_wid = self.reader.idx2wid[wids_batch[0]]
      true_WikiTitle = self.reader.wid2wikiTitle[true_wid]
      # idx_predwid = Idx in concated scores. 0:WidsBatch, [1:C] : CandsBatch
      idx_predwid = np.argmax(scores_with_cprob, axis=0)
      if idx_predwid == 0:
        pred_wid = true_wid
      else:
        pred_wid = self.reader.idx2wid[cands_batch[0][idx_predwid-1]]
      pred_WikiTitle = self.reader.wid2wikiTitle[pred_wid]
      print("True WikiTitle: {}".format(true_WikiTitle))
      print("Pred WikiTitle: {}".format(pred_WikiTitle))
    #endfor


    print("Correct Crosswikis, True WID has max prob in candidates: %d" % correct_crosswikis)
    print("Total Number of Correct Predictions w/o cprob: %d" % correct_wo_cprob)
    print("Total Number of Correct Predictions w cprob: %d" % correct_w_cprob)
    print("No candidates in Crosswikis: %d" % count_not_in_crosswikis)
    print("Total Number of Predictions : %d" % total_preds)
    sys.exit()
  #end testing

  def softmax(self, scores):
    expc = np.exp(scores)
    sumc = np.sum(expc)
    softmax_out = expc/sumc
    return softmax_out

  def print_all_variables(self):
    print("All Variables in the graph : ")
    self.print_variables_in_collection(tf.all_variables())

  def print_trainable_variables(self):
    print("All Trainable variables in the graph : ")
    self.print_variables_in_collection(tf.trainable_variables())

  def print_variables_in_collection(self, list_vars):
    print("Variables in list: ")
    for var in list_vars:
      print("  %s" % var.name)



