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

start_word = "<s>"
end_word = "<eos>"

class Mention(object):
  def __init__(self, mention_line):
    ''' mention_line : Is the string line stored for each mention
    mid wid wikititle start_token end_token surface tokenized_sentence all_types
    '''
    mention_line = mention_line.strip()
    split = mention_line.split("\t")
    (self.mid, self.wid, self.wikititle) = split[0:3]
    self.start_token = int(split[3]) + 1
    self.end_token = int(split[4]) + 1
    self.surface = split[5]
    self.sent_tokens = [start_word]
    self.sent_tokens.extend(split[6].split(" "))
    self.sent_tokens.append(end_word)
    self.types = split[7].split(" ")
    assert self.end_token <= (len(self.sent_tokens) - 1), "Line : %s" % mention_line
  #enddef
#endclass

class TypeStats(object):
  def __init__(self, train_mentions_dir, val_mentions_file,
               val_cold_mentions_file, label_vocab_pkl):

    print("[#] Loading label vocab ... ")
    (self.label2idx, self.idx2label) = load(label_vocab_pkl)
    self.num_labels = len(self.idx2label)
    print("[#] Label vocab loaded. Number of labels : {}".format(self.num_labels))

    print("[#] Training Mentions Dir : {}".format(train_mentions_dir))
    self.tr_mens_dir = train_mentions_dir
    self.tr_mens_files = self.get_mention_files(self.tr_mens_dir)
    #self.tr_mens_files = ["train.mens.5"]
    self.num_tr_mens_files = len(self.tr_mens_files)
    print("[#] Training Mention Files : {} files".format(self.num_tr_mens_files))

    print("[#] Validation Mentions File : {}".format(val_mentions_file))
    print("[#] Cold Validation Mentions File : {}".format(val_cold_mentions_file))

    print("[#] Pre-loading validation mentions ... ")
    self.val_mentions = self._make_mentions_from_file(val_mentions_file)
    self.cold_val_mentions = self._make_mentions_from_file(val_cold_mentions_file)
    self.num_val_mens = len(self.val_mentions)
    self.num_cold_val_mens = len(self.cold_val_mentions)
    print( "[#] Validation Mentions : {}, Cold Validation Mentions : {}".format(
          self.num_val_mens, self.num_cold_val_mens))

    print("\n[#] LOADING COMPLETE:")
    print("[1] Train mention file \n[2] Validation mentions \n"
          "[3] Label Set")

  #*******************      END __init__      *********************************

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

  def _load_mentions_from_file(self, mens_dir, mens_file):
    file = os.path.join(mens_dir, mens_file)
    mens = self._make_mentions_from_file(file)
    return mens

  def typestats(self):
    print("Computing Training Stats")
    total_tr_types = 0
    tr_mens = 0
    tr_type_set = set()
    for tr_file in self.tr_mens_files:
      print("Reading file : {}".format(tr_file))
      mens = self._load_mentions_from_file(self.tr_mens_dir, tr_file)
      for m in mens:
        tr_type_set.update(m.types)
        tr_mens += 1
        total_tr_types += len(m.types)
    tr_types_per_men = float(total_tr_types)/float(tr_mens)

    print("Computing Val Stats")
    total_v_types = 0
    v_mens = 0
    v_type_set = set()
    for m in self.val_mentions:
      v_type_set.update(m.types)
      v_mens += 1
      total_v_types += len(m.types)
    v_types_per_men = float(total_v_types)/float(v_mens)


    print("Computing Cold Val Stats")
    total_cv_types = 0
    cv_mens = 0
    cv_type_set = set()
    for m in self.cold_val_mentions:
      cv_type_set.update(m.types)
      cv_mens += 1
      total_cv_types += len(m.types)
    cv_types_per_men = float(total_cv_types)/float(cv_mens)

    print("Total Training Mentions : {}, Types per mention : {} "
          "Train Type Set Size : {}".format(tr_mens, tr_types_per_men,
                                            len(tr_type_set)))

    print("Total Val Mentions : {}, Types per mention : {} "
          "Val Type Set Size : {}".format(v_mens, v_types_per_men,
                                            len(v_type_set)))

    print("Total Cold Val Mentions : {}, Types per mention : {} "
          "Cold Val Type Set Size : {}".format(cv_mens, cv_types_per_men,
                                            len(cv_type_set)))

  #enddef

if __name__ == '__main__':
  batch_size = 1000
  num_batch = 1000
  b = TypeStats(
    train_mentions_dir="/save/ngupta19/wikipedia/wiki_mentions/train",
    val_mentions_file="/save/ngupta19/wikipedia/wiki_mentions/val/val.mens",
    val_cold_mentions_file="/save/ngupta19/wikipedia/wiki_mentions/val/val.single.mens",
    label_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/label_vocab.pkl")

  b.typestats()