import sys
import os
import time
import pickle
import numpy as np

def entity_set(filepath):
	enset = set()
	with open(filepath, 'r') as f:
		entities = f.read().strip().split("\n")
	enset = set(entities)
	print("Num of entities  = {}".format(len(enset)))
	return enset

def fun(s):
	s` += "n"

if __name__ == '__main__':
	train_set = entity_set("/save/ngupta19/wikipedia/figer/train_entities")
	val_set = entity_set("/save/ngupta19/wikipedia/figer/val_entities")
	interset = val_set.intersection(train_set)
	print("Num of entities in intersection = {}".format(len(interset)))

	s = "laskhfsd"
	fun(s)
	print(s)


