### Place to store helper functions that load and process the
### text data.
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

# adds data to a list for training or testing. Could add an
# even amount or uneven amount based on inputs.
def add_to_data(data_set, tumor_max, normal_max):
	normal_data = open(text_file_normal, "r")
	tumor_data = open(text_file_tumor, "r")
	tumors = 0
	normals = 0
	while tumors < tumor_max:
		next_line = tumor_data.readline().split()
		data_set += [(next_line[1:-1], next_line[-1])]
		tumors += 1
	while normals < normal_max:
		next_line = normal_data.readline().split()
		data_set += [(next_line[1:-1], next_line[-1])]
		normals += 1
	normal_data.close()
	tumor_data.close()

# to be used for determining output of NN,
# essentially our sigmoid function but other option is possible
label_to_ix = {"Tumor": 0, "Normal": 1}
def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])

# transforms a the gene list into a tensor for our NN
def make_gene_vector(file_line):
	data = list(map(float, file_line))
	vec = torch.FloatTensor(data)
	return vec.view(1, -1)

# makes a dictionary for gene index to be able to connect nodes
def gene_dict():
	f = open(text_data, "r")
	gene_names = f.readline().split()
	f.close()

	gene_to_index = {}
	for index, gene_name in enumerate(gene_names[1:-1]):
		if gene_name not in gene_to_index:
			gene_to_index[gene_name] = [index]
	else:
		gene_to_index[gene_name] = gene_to_index[gene_name] +[index]
	return gene_to_index

# import hallmark gene data
def import_gene_groups():
	f = open(text_gene_groups, "r")
	gene_groups = []
	for line in f:
		gene_data = line.split()
		gene_groups.append(gene_data[2:])
	f.close()
	return gene_groups

def get_gene_indicies(gene_group, gene_indexer):
	f = open("text_files/gene_pairs.pickle", "rb")
	alternate_names = pickle.load(f)
	f.close()
	indices = []
	for gene in gene_group:
		try:
			if gene in alternate_names.keys():
				gene = alternate_names[gene]
			indices += gene_indexer[gene]
		except:
			pass
	indices.sort()
	return indices


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

	if str(model) == "Zero-weights":
		if debug:
			print("Doing some calculations")
		model.make_mask(gene_groups)
		model.set_weights()

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
			target = make_target(label, label_to_ix)
			
			# get probabilities from instance
			output = model(gene_vec)

			# apply learning to the model based on the instance
			loss = loss_function(output, target)
			loss.backward()
			if str(model) == "Zero-weights": model.set_weights() 
			optimizer.step()

	if debug:
		print("\nSaving the model to file")
	torch.save(model.state_dict(), stored_dict_locs[str(model)])
	end_time = time.monotonic()
	if debug:
		print("Runtime:", timedelta(seconds=end_time - start_time), "\n")
