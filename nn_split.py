'''
i
 neural network where the structure of the network is 
 explicitely defined and the weights are not made zero

 An alternate beginning of the full project
 a lot of copied code from nn_partial_links.py that 
 will be edited to implement the above feature

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import time
from datetime import timedelta
import random
import copy
import pickle
from tqdm import tqdm

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
		# data_set will be tuple containing (list of genes, tumor/normal)
		data_set += [(next_line[1:-1], next_line[-1])]
		tumors += 1
	while normals < normal_max:
		next_line = normal_data.readline().split()
		data_set += [(next_line[1:-1], next_line[-1])]
		normals += 1
	normal_data.close()
	tumor_data.close()

# to be used for determining output of NN, essentially our sigmoid function
label_to_ix = {"Tumor": 0, "Normal": 1}
def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])

# transforms a the gene list into a tensor for our NN
# input should not include sample name nor tumor/normal identification
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
			# some genes are found in multiple locations
			gene_to_index[gene_name] = gene_to_index[gene_name] +[index]
	return gene_to_index

# import gene group data
def import_gene_groups():
	f = open(text_gene_groups, "r")
	gene_groups = []
	for line in f:
		gene_groups.append(line.split()[2:]) # first two points are unnecessary
	f.close()
	return gene_groups

# get the indexes of genes to be able to grab data based on 
# gene groups and build network topology
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

from operator import itemgetter

def split_data(input_vector, gene_group_indicies):
	data = input_vector.tolist()[0]
	#split = [list(map(data.__getitem__, group_indices)) for group_indices in gene_group_indicies]
	split = []
	for group_indices in gene_group_indicies:
		try:
			trimmed_data = list(itemgetter(*group_indices)(data))
			split.append(trimmed_data)
		except:
			split.append([])
	return split

# NN class 
class NN_split(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(NN_split, self).__init__()
		self.gene_group_indicies = []
		fc = []  
		gene_indexes = gene_dict()
		gene_groups = import_gene_groups()
		for i, gene_group in enumerate(gene_groups):
			if test_behavior and i in weights_to_test:
				fc.append(nn.Linear(1,1))
				fc[-1].weight = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)
				fc[-1].bias = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)
				self.gene_group_indicies.append([])
			else:
				group_indices = get_gene_indicies(gene_group, gene_indexes)
				self.gene_group_indicies.append(group_indices)
				# creates linear layers that has input
				# of gene group size and outputs a 
				# tensor of size 1
				fc.append(nn.Linear(len(group_indices), 1))
		self.linears = nn.ModuleList(fc)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size - len(weights_to_test), 2)

	def forward(self, input_vector):
		hidden = [] # list of tensors
		# here we must split the input_vector and active it
		# using the specific linear function it belongs to
		processed_data = split_data(input_vector, self.gene_group_indicies)
		hidden = [self.linears[index](torch.FloatTensor(processed_data[index])) for index in range(len(self.gene_group_indicies)) if index not in weights_to_test]
		# concatenate all the linear layers to make a 
		# tensor with size of gene groups
		hidden = torch.stack(hidden, 1)
		out = self.relu(hidden)
		# now take the output as normal
		out = self.fc2(out)
		out = F.log_softmax(out, dim=1)
		return out

	def __str__(self):
		return "Split"

def train_split_model():
	start_time = time.monotonic()

	### Hyperparameters
	hidden_size = len(import_gene_groups()) 
	
	# terminal message to track work
	if debug:
		print("Building the split model trained on the", mode, 
			tumor_data_size, "Tumor and", 
			normal_data_size, "Normal samples.", flush=True)
		print("Hyperparameters:", 
			num_epochs, "epochs,", 
			hidden_size, "neurons in the hidden layer,", 
			learning_rate, "learning rate.", flush=True)

	model = NN_split(hidden_size, output_size)
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
			instance, label = training_data[i]
			
			# erase gradients from previous run
			model.zero_grad()
			
			gene_vec = make_gene_vector(instance)
			target = make_target(label, label_to_ix)

			# get probabilities from instance
			output = model(gene_vec)

			# apply learning to the model based on the instance
			loss = loss_function(output, target)
			loss.backward()
			optimizer.step()

	if debug:
		print()
		print("Saving the model to file")
	torch.save(model.state_dict(), stored_dict_locs[str(model)])
	end_time = time.monotonic()
	if debug:
		print("Runtime:", timedelta(seconds=end_time - start_time))
		print()

if __name__ == "__main__":
	train_split_model()
