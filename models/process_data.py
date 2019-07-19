### Place to store helper functions that load and process the text data
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import sys
import time
from settings import *
from datetime import timedelta
import random
import pickle
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from collections import defaultdict

# adds data to a list for training or testing. Could add an
# even amount or uneven amount based on inputs.
def add_to_data(data_set, text_files):
	for text_file in text_files:
		with open(text_file, "r") as f:
			for line in f:
				line = line.split()
				data_set += [(line[1:-1], line[-1])]

def load_data(mode):
	dirs = {"train": train_dir, "test": test_dir}
	if debug: 
		print("Loading", mode, "data")
	data = []
	text_files = [dirs[mode] + data + "_" + mode + "_" + i + "_samples.txt" for i in ["normal", "tumor"]]
	add_to_data(data, text_files)
	return data

# to be used for determining output of NN,
# essentially our sigmoid function but other option is possible
label_to_ix = {"Tumor": 0, "Normal": 1}
def make_expected(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])

# transforms a the gene list into a tensor for our NN
def make_gene_vector(input_sample):
	processed = list(map(float, input_sample))
	vec = torch.FloatTensor(processed)
	return vec.view(1, -1)

# makes a dictionary for gene index to be able to connect nodes
def gene_dict():
	text_data = "text_files/" + data + "_gene_names.txt"
	with open(text_data, "r") as f:
		gene_names = f.readline().split()

	gene_to_index = defaultdict(list)
	for index, key in enumerate(gene_names[1:-1]):
		gene_to_index[key].append(index)
	return dict(gene_to_index)

# import hallmark gene data
def import_gene_groups():
	with open(text_gene_groups, "r") as f:
		gene_groups = [line.split()[2:] for line in f]
	return gene_groups

def get_gene_indicies(gene_group, gene_indexes):
	with open(gene_pairs_loc, "rb") as f:
		alternate_names = pickle.load(f)
	indices = []
	for gene in gene_group:
		try:
			if gene in alternate_names.keys():
				gene = alternate_names[gene]
			indices += gene_indexes[gene]
		except:
			pass
	indices.sort()
	return indices

def set_starting_seed(hidden_size=hidden_size):
	test_layer = nn.Linear(input_size, hidden_size)
	test_layer.weight.data.normal_(0.0, 1/(input_size**(0.5)))
	with open(starting_seed_loc, "wb") as f:
		pickle.dump(test_layer.weight.data, f) 

def get_starting_seed():
	with open(starting_seed_loc, "rb") as f:
		starting_seed = pickle.load(f)
	return starting_seed

def train_model(model, training_data):
	start_time = time.monotonic()
	# terminal message to track work
	if debug:
		print("Training", str(model), "model on the", 
			data, "dataset with",
			tumor_data_size, "Tumor samples and",
			normal_data_size, "Normal samples.")
		print("Hyperparameters:",
			num_epochs, "epochs,",
			hidden_size, "neurons in the hidden layer,",
			learning_rate, "learning rate.")

	model = model.train()
	loss_function = nn.NLLLoss()
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

	# train the model
	if debug:
		print("Training the model")
	for epoch in tqdm(range(num_epochs), disable=not debug):
		random.shuffle(training_data)
		for i in tqdm(range(len(training_data)), disable=not debug):
			# erase gradients from previous run
			instance, label = training_data[i]
			model.zero_grad()

			gene_vec = make_gene_vector(instance)
			expected = make_expected(label, label_to_ix)
			
			# get probabilities from instance
			output = model(gene_vec)

			# apply learning to the model based on the instance
			loss = loss_function(output, expected)
			loss.backward()
			optimizer.step()

	if debug:
		print("\nSaving the model to file")
	torch.save(model.state_dict(), stored_dict_locs[str(model)])
	end_time = time.monotonic()
	if debug:
		print("Runtime:", timedelta(seconds=end_time - start_time), "\n")
