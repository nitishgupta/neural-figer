import re
import os
import gc
import sys
import math
import pickle
import random
import unicodedata
import collections
import numpy as np
import time

def save(fname, obj):
  with open(fname, 'wb') as f:
    pickle.dump(obj, f)

def load(fname):
  with open(fname, 'rb') as f:
    return pickle.load(f)


class TestingDataReader(object):
  def __init__(self, test_file, word_vocab_pkl, label_vocab_pkl,
               word2vec_bin_gz, batch_size):
    self.unk_word = '<unk_word>' # In tune with word2vec
    self.unk_wid = "<unk_wid>"
    self.tr_sup = 'tr_sup'
    self.tr_unsup = 'tr_unsup'
    self.sanity_checks(test_file, word2vec_bin_gz, batch_size)

    # WID VOCAB
    assert os.path.exists(word_vocab_pkl), "Word Vocab does not exist"
    print("[#] Loading word vocab ... ")
    (self.word2idx, self.idx2word) = load(word_vocab_pkl)
    self.num_words = len(self.idx2word)
    print(" [#] Word vocab loaded. Size of vocab : {}".format(self.num_words))

    assert os.path.exists(label_vocab_pkl), "Label Vocab does not exist"
    print("[#] Loading label vocab ... ")
    (self.label2idx, self.idx2label) = load(label_vocab_pkl)
    self.num_labels = len(self.idx2label)
    print(" [#] Label vocab loaded. Number of labels : {}".format(self.num_labels))

    '''
    # Word2Vec Gensim Model
    self.word2vec_model = Word2Vec(word2vec_bin_gz=word2vec_bin_gz,
                                   use_shelve=False)
    self.w2v_dim = self.word2vec_model.dim
    self.unk_vector = self.word2vec_model.get_vector('unk')
    '''
    '''
    # Crosswikis dictionary
    stime = time.time()
    print("[#] Loading normalized crosswikis dictionary ... ")
    self.crosswikis_dict = load(crosswikis_norm_pkl)
    ttime = (time.time() - stime)/60.0
    print(" [#] Crosswikis dictionary loaded!. Time Take : %2.4f mins" % ttime)
    self.cwikis_candidate_thresh = cwikis_candidate_thresh
    print("[#] Crosswikis Candidate Threshold: %d" % self.cwikis_candidate_thresh)
    '''

    self.batch_size = batch_size
    print("[#] Batch Size: %d" % self.batch_size)

    print("[#] Loading test mentions ... ")
    self.epochs = 0
    self.mens_idx = 0
    with open(test_file, 'r') as f:
      self.mentions = f.read().strip().split("\n")
      random.shuffle(self.mentions)
    self.num_mens = len(self.mentions)
    print(" [#] Test mentions loaded. Num of mentions: {}".format(self.num_mens))

    print("\n[#] LOADING COMPLETE:")
    print("[1] Train mention file \n[2] Validation mentions \n"
          "[3] Word Vocab \n[4] Label Set")

  #*******************      END __init__      *********************************

  def sanity_checks(self, test_file, word2vec_bin_gz, batch_size):
    assert os.path.exists(test_file), "Training File Missing!!"
    assert os.path.exists(word2vec_bin_gz), "Word2vec bin.gz missing"
    assert batch_size > 0, "Batch Size < 0"
  #end_sanity checks

  def reset_validation(self):
    self.val_mens_idx = 0
    self.val_epochs = 0

  def _read_mention(self):
    # Read test mention
    # If all mentions read
    if self.mens_idx == self.num_mens:
      random.shuffle(self.mentions)
      self.mens_idx = 0
      self.epochs += 1
    #endif
    mention = self.mentions[self.mens_idx]
    self.mens_idx += 1
    return mention
  #enddef

  def _getLnrm(self, arg):
    """Normalizes the given arg by stripping it of diacritics, lowercasing, and
    removing all non-alphanumeric characters.
    """
    arg = ''.join([
      c for c in unicodedata.normalize('NFD', arg)
      if unicodedata.category(c) != 'Mn'
    ])
    arg = arg.lower()
    arg = ''.join([
      c for c in arg
      if c in set('abcdefghijklmnopqrstuvwxyz0123456789')
    ])
    return arg

  def _next_batch(self):
    ''' Data : wikititle \t mid \t wid \t start \t end \t tokens \t labels
    start and end are inclusive
    '''
    # Sentence     = s1 ... m1 ... mN, ... sN.
    # Left Batch   = s1 ... m1 ... mN
    # Right Batch  = sN ... mN ... m1
    (left_batch, right_batch) = ([], [])

    # Labels : Vector of 0s and 1s of size = number of labels = 113
    labels_batch = np.zeros([self.batch_size, self.num_labels])

    while len(left_batch) < self.batch_size:
      batch_el = len(left_batch)
      mention = self._read_mention().strip()
      while (mention == ""):
        mention = self._read_mention().strip()

      ssplit = mention.split("\t")
      assert len(ssplit) == 7
      start = int(ssplit[3])
      end = int(ssplit[4])
      tokens = ssplit[5].split(" ")
      assert end <= (len(tokens) - 1), "End and NumTokens error"
      labels = ssplit[6].split(" ")

      for label in labels:
        labelidx = self.label2idx[label]
        labels_batch[batch_el][labelidx] = 1.0
      #labels

      # Left and Right context includes mention surface
      #left_tokens = tokens[0:end+1]
      #right_tokens = tokens[start:][::-1]

      # Strict left and right context
      left_tokens = tokens[0:start]
      right_tokens = tokens[end+1:][::-1]

      left_idxs = [self.convert_word2idx(word) for word in left_tokens]
      right_idxs = [self.convert_word2idx(word) for word in right_tokens]

      left_batch.append(left_idxs)
      right_batch.append(right_idxs)

    return (left_batch, right_batch, labels_batch)
  #enddef

  def pad_batch(self, batch):
    unk_word_idx = self.word2idx[self.unk_word]
    lengths = [len(i) for i in batch]
    max_length = max(lengths)
    for i in range(0, len(batch)):
      batch[i].extend([unk_word_idx]*(max_length - lengths[i]))
    #endfor
    return (batch, lengths)

  def _next_padded_batch(self):
    (left_batch, right_batch, labels_batch) = self._next_batch()
    (left_batch, left_lengths) = self.pad_batch(left_batch)
    (right_batch, right_lengths) = self.pad_batch(right_batch)

    return (left_batch, left_lengths, right_batch, right_lengths, labels_batch)
  #enddef

  def convert_word2idx(self, word):
    if word in self.word2idx:
      return self.word2idx[word]
    else:
      return self.word2idx[self.unk_word]
  #enddef

  def next_test_batch(self):
    return self._next_padded_batch()

if __name__ == '__main__':
  batch_size = 5
  b = TestingDataReader(
    test_file="/save/ngupta19/wikipedia/figer/test.data",
    word_vocab_pkl="/save/ngupta19/wikipedia/figer/vocab/word_vocab.pkl",
    label_vocab_pkl="/save/ngupta19/wikipedia/figer/vocab/label_vocab.pkl",
    word2vec_bin_gz="/save/ngupta19/word2vec/GoogleNews-vectors-negative300.bin.gz",
    batch_size=batch_size)

  stime = time.time()
  num_batch = 1000
  i = 0
  while b.epochs < 1:
  #for i in range(0, num_batch):
    (left_batch, left_lengths,
     right_batch, right_lengths, labels_batch) = b.next_test_batch()
    if i%100 == 0:
      #print(labels_batch)
      etime = time.time()
      t=etime-stime
      print("{} done. Time taken : {} seconds".format(i, t))
    i += 1
  #endfor
  etime = time.time()
  t=etime-stime
  print("Total time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, t))