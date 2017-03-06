import os
import sys
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=7)

from utils import pp
from readers.training_reader import TrainingDataReader
from models.figer_model.figer_model import FigerModel

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_integer("max_steps", 40000, "Maximum of iteration [450000]")
flags.DEFINE_integer("pretraining_steps", 25000, "Number of steps to run pretraining")
flags.DEFINE_string("model", "figer", "The name of model [nvdm, nasm]")
flags.DEFINE_string("dataset", "figer", "The name of dataset [ptb]")
flags.DEFINE_string("checkpoint_dir", "/save/ngupta19/checkpoint", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_integer("batch_size", 100, "Batch Size for training and testing")

flags.DEFINE_integer("word_embed_dim", 300, "Word Embedding Size")
flags.DEFINE_integer("context_encoded_dim", 300, "Context Encoded Dim")

flags.DEFINE_integer("context_encoder_num_layers", 1, "Num of Layers in context encoder network")
flags.DEFINE_integer("context_encoder_lstmsize", 300, "Size of context encoder hidden layer")
flags.DEFINE_float("reg_constant", 0.00, "Regularization constant for NN weight regularization")
flags.DEFINE_float("dropout_keep_prob", 0.6, "Dropout Keep Probability")
flags.DEFINE_boolean("decoder_bool", True, "Decoder bool")
flags.DEFINE_string("mode", 'tr_sup', "Mode to run")
flags.DEFINE_boolean("strict_context", True, "Strict Context exludes mention surface")
flags.DEFINE_boolean("pretrain_wordembed", False, "Use Word2Vec Embeddings")
flags.DEFINE_string("optimizer", 'optim', "Optimizer to use. adagrad, adadelta or adam")


FLAGS = flags.FLAGS

def get_test_dataset_paths(dataset):
  if dataset == 'MSNBC' or dataset == 'msnbc':
    mentions_test_file="/save/ngupta19/datasets/MSNBC/mentions.txt"
    test_docs_dir="/save/ngupta19/datasets/MSNBC/docs"
    test_links_dir="/save/ngupta19/datasets/MSNBC/links"
  if dataset == 'ACE' or dataset == 'ace':
    mentions_test_file="/save/ngupta19/datasets/ACE/mentions.txt"
    test_docs_dir="/save/ngupta19/datasets/ACE/docs"
    test_links_dir="/save/ngupta19/datasets/ACE/links"
  if dataset == 'AIDAtr' or dataset == 'aidatr':
    mentions_test_file="/save/ngupta19/datasets/AIDA/mentions_train_nonnme.txt"
    test_docs_dir="/save/ngupta19/datasets/AIDA/docs"
    test_links_dir="/save/ngupta19/datasets/AIDA/links"
  if dataset == 'AIDAtest' or dataset == 'aidatest':
    mentions_test_file="/save/ngupta19/datasets/AIDA/mentions_test_nonnme.txt"
    test_docs_dir="/save/ngupta19/datasets/AIDA/docs"
    test_links_dir="/save/ngupta19/datasets/AIDA/links"
  if dataset == 'AIDAdev' or dataset == 'aidadev':
    mentions_test_file="/save/ngupta19/datasets/AIDA/mentions_dev_nonnme.txt"
    test_docs_dir="/save/ngupta19/datasets/AIDA/docs"
    test_links_dir="/save/ngupta19/datasets/AIDA/links"
  if dataset == 'AIDAtrain' or dataset == 'aidatrain':
    mentions_test_file="/save/ngupta19/datasets/AIDA/mentions_train_nonnme.txt"
    test_docs_dir="/save/ngupta19/datasets/AIDA/docs"
    test_links_dir="/save/ngupta19/datasets/AIDA/links"
  if dataset == 'WikiTrain' or dataset == 'wikitrain':
    mentions_test_file="/save/ngupta19/datasets/WIKI/mentions_train.txt"
    test_docs_dir="/save/ngupta19/datasets/WIKI/docs"
    test_links_dir="/save/ngupta19/datasets/WIKI/links"
  if dataset == 'WikiTest' or dataset == 'wikitest':
    mentions_test_file="/save/ngupta19/datasets/WIKI/mentions_test.txt"
    test_docs_dir="/save/ngupta19/datasets/WIKI/docs"
    test_links_dir="/save/ngupta19/datasets/WIKI/links"

  return (mentions_test_file, test_docs_dir, test_links_dir)

def optimizer_checks(FLAGS):
  if (FLAGS.optimizer != 'adagrad' and
      FLAGS.optimizer != 'adadelta' and
      FLAGS.optimizer != 'adam' and
      FLAGS.optimizer != 'sgd' and
      FLAGS.optimizer != 'momentum'):
    print("*** Optimizer not defined. *** ")
    sys.exit(0)

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  test_dataset = 'aidatrain'
  (mentions_test_file,
   test_docs_dir,
   test_links_dir) = get_test_dataset_paths(test_dataset)

  train_dir = "/save/ngupta19/wikipedia/wiki_mentions/train"
  val_file = "/save/ngupta19/wikipedia/wiki_mentions/val/val.mens"
  cold_val_file = "/save/ngupta19/wikipedia/wiki_mentions/val/val.single.mens"
  ace_test_file = "/save/ngupta19/datasets/ACE/mentions_inkb.txt"
  aida_train_file="/save/ngupta19/datasets/AIDA/inkb_mentions/mentions_train_inkb.txt"
  aida_dev_file="/save/ngupta19/datasets/AIDA/inkb_mentions/mentions_dev_inkb.txt"
  aida_test_file="/save/ngupta19/datasets/AIDA/inkb_mentions/mentions_test_inkb.txt"
  word_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/figer/word_vocab.pkl"
  label_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/figer/label_vocab.pkl"
  word2vec_bin_gz="/save/ngupta19/word2vec/GoogleNews-vectors-negative300.bin.gz"

  optimizer_checks(FLAGS)

  if FLAGS.mode == 'tr_sup' or FLAGS.mode == 'tr_unsup':
    reader = TrainingDataReader(
      train_mentions_dir=train_dir,
      val_mentions_file=val_file,
      val_cold_mentions_file=cold_val_file,
      word_vocab_pkl=word_vocab_pkl,
      label_vocab_pkl=label_vocab_pkl,
      word2vec_bin_gz=word2vec_bin_gz,
      batch_size=FLAGS.batch_size,
      strict_context=FLAGS.strict_context,
      pretrain_wordembed=FLAGS.pretrain_wordembed)
    model_mode = 'train'  # Needed for batch normalization
  elif FLAGS.mode == 'test':
    reader = TrainingDataReader(
      train_mentions_dir=train_dir,
      val_mentions_file=ace_test_file,
      val_cold_mentions_file=aida_train_file,
      word_vocab_pkl=word_vocab_pkl,
      label_vocab_pkl=label_vocab_pkl,
      word2vec_bin_gz=word2vec_bin_gz,
      batch_size=FLAGS.batch_size,
      strict_context=FLAGS.strict_context,
      pretrain_wordembed=FLAGS.pretrain_wordembed)
    model_mode = 'test'  # Needed for batch normalization
    FLAGS.dropout_keep_prob = 1.0
  else:
    print("MODE in FLAGS is incorrect : {}".format(FLAGS.mode))
    sys.exit()

  config_proto = tf.ConfigProto()
  config_proto.allow_soft_placement = True
  config_proto.gpu_options.allow_growth=True
  #config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
  # intra_op_parallelism_threads=NUM_THREADS
  #config_proto.log_device_placement=True
  sess = tf.Session(config=config_proto)

  with sess.as_default():
    model = FigerModel(sess=sess, reader=reader, dataset=FLAGS.dataset,
              max_steps=FLAGS.max_steps,
              pretrain_max_steps=FLAGS.pretraining_steps,
              word_embed_dim=FLAGS.word_embed_dim,
              context_encoded_dim=FLAGS.context_encoded_dim,
              context_encoder_num_layers=FLAGS.context_encoder_num_layers,
              context_encoder_lstmsize=FLAGS.context_encoder_lstmsize,
              learning_rate=FLAGS.learning_rate,
              dropout_keep_prob=FLAGS.dropout_keep_prob,
              reg_constant=FLAGS.reg_constant,
              checkpoint_dir=FLAGS.checkpoint_dir,
              optimizer=FLAGS.optimizer,
              mode=model_mode,
              strict=FLAGS.strict_context,
              pretrain_word_embed=FLAGS.pretrain_wordembed)

    if FLAGS.mode=='test':
      print("Doing inference")
      model.testing()
    elif FLAGS.mode=='tr_sup':
      model.training()
    elif FLAGS.mode=='tr_unsup':
      model.training()
    else:
      print("WRONG MODE!")
      sys.exit(0)



if __name__ == '__main__':
  tf.app.run()

