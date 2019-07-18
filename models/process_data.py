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
def add_to_data(data_set, tumor_max, normal_max):
	normals = 0	
	with open(text_file_normal, "r") as normal_data:
		while normals < normal_max:
			line = normal_data.readline().split()
			data_set += [(line[1:-1], line[-1])]
			normals += 1

	tumors = 0
	with open(text_file_tumor, "r") as tumor_data:
		while tumors < tumor_max:
			line = tumor_data.readline().split()
			data_set += [(line[1:-1], line[-1])]
			tumors += 1


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

def set_starting_seed():
	test_layer = nn.Linear(input_size, hidden_size)
	test_layer.weight.data.normal_(0.0, 1/(input_size**(0.5)))
	with open(starting_seed_loc, "wb") as f:
		pickle.dump(test_layer.weight.data, f) 

def get_starting_seed():
	with open(starting_seed_loc, "rb") as f:
		starting_seed = pickle.load(f)
	return starting_seed

def train_model(model):
	start_time = time.monotonic()

	### Hyperparameters
	gene_groups = import_gene_groups()
	hidden_size = len(gene_groups)

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
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

	if debug:
		print("Loading the data", end='', flush=True)
	training_data = []
	add_to_data(training_data, tumor_data_size, normal_data_size)
	sys.stdout.write("\r")
	sys.stdout.flush()
	if debug:
		print("Loaded the data ", flush=True)

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
