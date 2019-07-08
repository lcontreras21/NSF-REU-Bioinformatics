# neural network where the hidden layer is connected 
# to the input based on gene groups. ie not all nodes
# are connected

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import time
from datetime import timedelta
import random
import pickle
from tqdm import tqdm
import numpy as np

# settings file that has what files to use and hyperparameters
from settings import *

# functions that I have used, plus new functions to read files.

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

def make_mask():
	gene_indexer = gene_dict()
	mask = np.array([[0] * len(training_data[0][0])] * len(gene_groups))
	for group_index, gene_group in enumerate(gene_groups):
		if test_behavior and group_index in weights_to_test:
			group_indices = []
		else:
			group_indices = get_gene_indicies(gene_group, gene_indexer)
		mask[group_index][group_indices] = 1
	mask = torch.FloatTensor(mask)
	return mask

# using gene groups data, set the weights 
# in the first layer to zero
# for the genes that are not in the group.
def set_weights(model, mask): 
	# we are interested in the first layer in model_state
	model.fc1.weight.data *= mask
	
# NN class 
class NN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(NN, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, output_size)
	def forward(self, input_vector):
		out = self.fc1(input_vector)
		out = self.relu(out)
		out = self.fc2(out)
		out = F.log_softmax(out, dim=1)
		return out
	def __str__(self):
		return "Zero-weights"


if __name__ == "__main__":
	start_time = time.monotonic()

	### Hyperparameters
	gene_groups = import_gene_groups()
	hidden_size = len(gene_groups) 
	
	# terminal message to track work
	if debug:
		print("Building the zero-weights model trained on the", mode, 
			tumor_data_size, "Tumor and", 
			normal_data_size, "Normal samples.")
		print("Hyperparameters:", 
			num_epochs, "epochs,", 
			hidden_size, "neurons in the hidden layer,", 
			learning_rate, "learning rate.")

	model = NN(input_size, hidden_size, output_size)
	model = model.train()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	
	if debug:
		print("Loading the data", end='', flush=True)
	training_data = []
	add_to_data(training_data, tumor_data_size, normal_data_size)
	sys.stdout.write("\r")
	sys.stdout.flush()
	if debug:
		print("Loaded the data ", flush=True)

	if debug:
		print("Doing some calculations")
	mask = make_mask()

	# set the starting weights to model the biology
	if debug:
		print("Setting the weights of the model")
	set_weights(model, mask)
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
			set_weights(model, mask)
			optimizer.step()
	
	if debug:
		print()
		print("Saving the model to file")
	torch.save(model.state_dict(), partial_dict)
	end_time = time.monotonic()
	if debug:
		print("Runtime:", timedelta(seconds=end_time - start_time))
		print()
