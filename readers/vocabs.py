import re
import os
import gc
import sys
import math
import time
import pickle
import random
import unicodedata
import collections
import numpy as np

def save(fname, obj):
	with open(fname, 'wb') as f:
		pickle.dump(obj, f)

def load(fname):
	with open(fname, 'rb') as f:
		return pickle.load(f)

def _make_mentions_from_file(mens_file):
	with open(mens_file, 'r') as f:
		mention_lines = f.read().strip().split("\n")
	mentions = []
	for line in mention_lines:
		mentions.append(Mention(line))
	return mentions
#enddef

def get_mention_files(mentions_dir):
	mention_files = []
	for (dirpath, dirnames, filenames) in os.walk(mentions_dir):
		mention_files.extend(filenames)
		break
	#endfor
	return mention_files
#enddef

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

class VocabBuilder(object):
	def __init__(self, train_mentions_dir, val_mentions_dir, word_vocab_pkl,
							 label_vocab_pkl, word_threshold=5):
		self.unk_word = '<unk_word>' # In tune with word2vec
		self.unk_wid = '<unk_wid>'

		self.tr_mens_files = get_mention_files(train_mentions_dir)
		self.val_mens_files = get_mention_files(val_mentions_dir)

		tr_data_vocabs_exist = self.check_train_data_vocabs_exist(
			word_vocab_pkl, label_vocab_pkl)

		if not tr_data_vocabs_exist:
			print("All/Some Training Vocabs do not exist. Making ... ")
			self.make_training_data_vocabs(train_mentions_dir, self.tr_mens_files,
																		 word_vocab_pkl, label_vocab_pkl,
																		 word_threshold)
		else:
			print("All Vocabs Exist. Exiting.")
		#end-vocab-init

	def check_train_data_vocabs_exist(self, word_vocab_pkl, label_vocab_pkl):
		if (os.path.exists(word_vocab_pkl) and os.path.exists(label_vocab_pkl)):
			return True
		else:
			return False

	def add_to_vocab(self, element2idx, idx2element, element):
		if element not in element2idx:
			idx2element.append(element)
			element2idx[element] = len(idx2element) - 1

	def make_training_data_vocabs(self, tr_mens_dir, tr_mens_files,
																word_vocab_pkl, label_vocab_pkl,
																knwn_wid_vocab_pkl, threshold):

		print("Building training vocabs : ")
		word_count_dict = {}
		idx2word = [self.unk_word]
		word2idx = {self.unk_word:0}
		idx2label = []
		label2idx = {}

		files_done = 0
		for file in tr_mens_files:
			print("Files done : {}".format(files_done))
			mens_fpath = os.path.join(tr_mens_dir, file)
			mentions = _make_mentions_from_file(mens_file=mens_fpath)
			for mention in mentions:
				for typel in mention.types:
					self.add_to_vocab(element2idx=label2idx, idx2element=idx2label,
														element=typel)
				for token in mention.sent_tokens:
					if token not in word_count_dict:
						word_count_dict[token] = 0
					word_count_dict[token] = word_count_dict[token] + 1

			files_done += 1
		#all-files-processed
		# WORD VOCAB
		for word, count in word_count_dict.items():
			if count > threshold:
				self.add_to_vocab(element2idx=word2idx, idx2element=idx2word,
													element=word)

		print(" [#] Threhsolded word vocab. Word Vocab Size: {}".format(len(idx2word)))
		save(word_vocab_pkl, (word2idx, idx2word))
		print(" [#] Label Vocab Size: {}".format(len(idx2label)))
		save(label_vocab_pkl, (label2idx, idx2label))


if __name__ == '__main__':
	batch_size = 1000
	num_batch = 1000
	b = VocabBuilder(
		train_mentions_dir="/save/ngupta19/wikipedia/wiki_mentions/train",
		val_mentions_dir="/save/ngupta19/wikipedia/wiki_mentions/val",
		word_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/word_vocab.pkl",
		label_vocab_pkl="/save/ngupta19/wikipedia/wiki_mentions/vocab/label_vocab.pkl",
		word_threshold=5)