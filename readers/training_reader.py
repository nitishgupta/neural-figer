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

class Mention(object):
  def __init__(self, mention_line):
    ''' mention_line : Is the string line stored for each mention
    mid wid wikititle start_token end_token surface tokenized_sentence all_types
    '''
    mention_line = mention_line.strip()
    split = mention_line.split("\t")
    (self.mid, self.wid, self.wikititle) = split[0:3]
    self.start_token = int(split[3])
    self.end_token = int(split[4])
    self.surface = split[5]
    self.sent_tokens = split[6].split(" ")
    self.types = split[7].split(" ")
    #self.end_token = min(self.end_token, len(self.sent_tokens) - 1)
    assert self.end_token <= (len(self.sent_tokens) - 1), "Line : %s" % mention_line
  #enddef
#endclass

class TrainingDataReader(object):
  def __init__(self, train_mentions_dir, val_mentions_dir, word_vocab_pkl,
               label_vocab_pkl, word2vec_bin_gz, batch_size, word_threshold=None):
    self.unk_word = '<unk_word>' # In tune with word2vec
    self.unk_wid = "<unk_wid>"
    self.tr_sup = 'tr_sup'
    self.tr_unsup = 'tr_unsup'
    #self.sanity_checks(train_file, word2vec_bin_gz, batch_size)

    # WID VOCAB
    if not os.path.exists(word_vocab_pkl):
      assert word_threshold != None, "Word Threshold for Vocab making needed!"
      print("[#] Creating word vocab from training data. Threshold: {}".format(word_threshold))
      self.make_word_vocab(train_file, word_vocab_pkl, word_threshold)
    print("[#] Loading word vocab ... ")
    (self.word2idx, self.idx2word) = load(word_vocab_pkl)
    self.num_words = len(self.idx2word)
    print(" [#] Word vocab loaded. Size of vocab : {}".format(self.num_words))

    # Label Vocab
    if not os.path.exists(label_vocab_pkl):
      print("[#] Creating label vocab from training data ... ")
      self.make_label_vocab(train_file, label_vocab_pkl)
    print("[#] Loading label vocab ... ")
    (self.label2idx, self.idx2label) = load(label_vocab_pkl)
    self.num_labels = len(self.idx2label)
    print(" [#] Label vocab loaded. Number of labels : {}".format(self.num_labels))


    print("[#] Training Mentions Dir : {}".format(train_mentions_dir))
    self.tr_mens_dir = train_mentions_dir
    self.tr_mens_files = self.get_mention_files(self.tr_mens_dir)
    self.num_tr_mens_files = len(self.tr_mens_files)
    print(" [#] Training Mention Files : {} files".format(self.num_tr_mens_files))

    print("[#] Validation Mentions Dir : {}".format(val_mentions_dir))
    self.val_mens_dir = val_mentions_dir
    self.val_mens_files = self.get_mention_files(self.val_mens_dir)
    self.num_val_mens_files = len(self.val_mens_files)
    print(" [#] Validation Mention Files : {} files".format(self.num_val_mens_files))

    self.tr_mentions = []
    self.tr_men_idx = 0
    self.num_tr_mens = 0
    self.tr_fnum = 0
    self.tr_epochs = 0
    self.val_mentions = []
    self.val_men_idx = 0
    self.num_val_mens = 0
    self.val_fnum = 0
    self.val_epochs = 0

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

    print("\n[#] LOADING COMPLETE:")
    print("[1] Train mention file \n[2] Validation mentions \n"
          "[3] Word Vocab \n[4] Label Set")

  #*******************      END __init__      *********************************

  def sanity_checks(self, train_file, word2vec_bin_gz, batch_size):
    assert os.path.exists(train_file), "Training File Missing!!"
    assert os.path.exists(word2vec_bin_gz), "Word2vec bin.gz missing"
    assert batch_size > 0, "Batch Size < 0"
  #end_sanity checks

  def get_mention_files(self, mentions_dir):
    mention_files = []
    for (dirpath, dirnames, filenames) in os.walk(mentions_dir):
      mention_files.extend(filenames)
      break
    #endfor
    random.shuffle(mention_files)
    return mention_files
  #enddef

  def _make_mentions_from_file(self, mens_file):
    with open(mens_file, 'r') as f:
      mention_lines = f.read().strip().split("\n")
    mentions = []
    for line in mention_lines:
      mentions.append(Mention(line))
    return mentions
  #enddef

  def _load_mentions_from_file(self, mens_dir, mens_files, num_mens_files, fnum, epochs):
    if fnum == num_mens_files:
      fnum = 0
      epochs += 1
      random.shuffle(mens_files)
    file = os.path.join(mens_dir, mens_files[fnum])
    fnum += 1
    # self.tr_mens is a list of objects of Mention class
    mens = self._make_mentions_from_file(file)
    return (mens, mens_files, fnum, epochs)

  def load_mentions_from_file(self, data_idx=0):
    # data_idx : train = 0, validation = 1
    if data_idx==0 or data_idx=="tr":
      stime = time.time()
      (self.tr_mens, self.tr_mens_files,
       self.tr_fnum, self.tr_epochs) = self._load_mentions_from_file(
        self.tr_mens_dir, self.tr_mens_files, self.num_tr_mens_files,
        self.tr_fnum, self.tr_epochs)
      self.num_tr_mens = len(self.tr_mens)
      self.tr_men_idx = 0
      ttime = (time.time() - stime)/60.0
      print("Loaded tr mentions. Num of mentions : {}. Time : {:.2f} mins".format(
        self.num_tr_mens, ttime))
      print("File Number loaded : {}".format(self.tr_fnum))

    if data_idx==1 or data_idx=="val":
      stime = time.time()
      if self.num_val_mens_files == 1 and self.val_fnum == self.num_val_mens_files:
        self.val_epochs += 1
      else:
        (self.val_mens, self.val_mens_files,
         self.val_fnum, self.val_epochs) = self._load_mentions_from_file(
          self.val_mens_dir, self.val_mens_files, self.num_val_mens_files,
          self.val_fnum, self.val_epochs)

      self.num_val_mens = len(self.val_mens)
      self.val_men_idx = 0
      ttime = (time.time() - stime)/60.0
      #print("Loaded val mentions. Num of mentions : {}. Time : {:.2f} mins".format(
      #  self.num_val_mens, ttime))
      #print("File Number loaded : {}".format(self.val_fnum))
  #enddef

  def reset_validation(self):
    self.val_men_idx = 0
    self.val_epochs = 0
    self.val_fnum = 0
    self.num_val_mens = 0

  def _read_mention(self, data_type=0):
    # Read train mention
    if data_type == 0 or data_type=="tr":
      # If all mentions read or no ments in memory
      if self.tr_men_idx == self.num_tr_mens or self.num_tr_mens == 0:
        self.load_mentions_from_file(data_type)

      mention = self.tr_mens[self.tr_men_idx]
      self.tr_men_idx += 1
      return mention
    # Read val mention
    if data_type == 1 or data_type == "val":
      # If all mentions read or no mentions in memory
      if self.val_men_idx == self.num_val_mens or self.num_val_mens == 0:
        self.load_mentions_from_file(data_type)

      mention = self.val_mens[self.val_men_idx]
      self.val_men_idx += 1
      return mention

    print("Wrong data_type arg. Quitting ... ")
    sys.exit(0)
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

  def _next_batch(self, data_type):
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
      m = self._read_mention(data_type=data_type)

      start = m.start_token
      end = m.end_token

      for label in m.types:
        labelidx = self.label2idx[label]
        labels_batch[batch_el][labelidx] = 1.0
      #labels

      # Left and Right context includes mention surface
      left_tokens = m.sent_tokens[0:m.end_token+1]
      right_tokens = m.sent_tokens[m.start_token:][::-1]

      # Strict left and right context
      #left_tokens = tokens[0:start]
      #right_tokens = tokens[end+1:][::-1]

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

  def _next_padded_batch(self, data_type):
    (left_batch, right_batch, labels_batch) = self._next_batch(data_type=data_type)
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

  def next_train_batch(self):
    return self._next_padded_batch(data_type=0)

  def next_val_batch(self):
    return self._next_padded_batch(data_type=1)

  def add_to_vocab(self, element2idx, idx2element, element):
    if element not in element2idx:
      idx2element.append(element)
      element2idx[element] = len(idx2element) - 1

  def make_word_vocab(self, train_file, word_vocab_pkl, threshold):
    #enddef
    print(" [#] Loading training mentions ... ")
    with open(train_file, 'r') as f:
      mentions = f.read().strip().split("\n")
    num_mens = len(mentions)
    print(" [#] Training mentions loaded. Num of mentions: {}".format(num_mens))
    print(" [#] Starting to make word vocab ...")

    word_count_dict = {}
    # Making word-count dictionary
    for mention in mentions:
      ssplit = mention.split("\t")
      tokens = ssplit[5].split(" ")
      for token in tokens:
        if token not in word_count_dict:
          word_count_dict[token] = 0
        word_count_dict[token] = word_count_dict[token] + 1
      #endfor
    #end-processed all mentions
    print(" [#] Read all tokens. Number of unique tokens: {}".format(len(word_count_dict)))

    # WORD VOCAB
    idx2word = [self.unk_word]
    word2idx = {self.unk_word:0}
    for word, count in word_count_dict.items():
      if count > threshold:
        self.add_to_vocab(element2idx=word2idx, idx2element=idx2word,
                          element=word)
    #endfor
    print(" [#] Threhsolded word vocab. Word Vocab Size: {}".format(len(idx2word)))
    save(word_vocab_pkl, (word2idx, idx2word))
  #enddef

  def make_label_vocab(self, train_file, label_vocab_pkl):
    #enddef
    print("[#] Loading training mentions ... ")
    with open(train_file, 'r') as f:
      mentions = f.read().strip().split("\n")
    num_mens = len(mentions)
    print("[#] Training mentions loaded. Num of mentions: {}".format(num_mens))
    print("[#] Starting to make label set ...")

    idx2label = []
    label2idx = {}

    # Making word-count dictionary
    for mention in mentions:
      ssplit = mention.split("\t")
      labels = ssplit[6].split(" ")
      for label in labels:
        self.add_to_vocab(element2idx=label2idx, idx2element=idx2label,
                          element=label)
      #endfor
    #end-processed all mentions
    print(" [#] Label set made. Label Set Size: {}".format(len(idx2label)))
    save(label_vocab_pkl, (label2idx, idx2label))
  #enddef

if __name__ == '__main__':
  batch_size = 1000
  num_batch = 1000
  b = TrainingDataReader(
    train_mentions_dir="/save/ngupta19/wikipedia/wiki_mentions/train",
    val_mentions_dir="/save/ngupta19/wikipedia/wiki_mentions/val",
    word_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/word_vocab.pkl",
    label_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/label_vocab.pkl",
    word2vec_bin_gz="/save/ngupta19/word2vec/GoogleNews-vectors-negative300.bin.gz",
    batch_size=batch_size,
    word_threshold=1)

  stime = time.time()

  i = 0
  total_instances = 0
  while b.tr_epochs < 1 and b.val_epochs < 1:
  #for i in range(0, num_batch):
    (left_batch, left_lengths,
     right_batch, right_lengths, labels_batch) = b.next_val_batch()
    total_instances += len(left_batch)
    if i%100 == 0:
      #print(labels_batch)
      etime = time.time()
      t=etime-stime
      print("{} done. Time taken : {} seconds".format(i, t))
    i += 1
  #endfor
  etime = time.time()
  t=etime-stime
  print("Total Instances : {}".format(total_instances))
  print("Total time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, t))