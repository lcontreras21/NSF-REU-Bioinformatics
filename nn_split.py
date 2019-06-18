'''
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

# settings file that has what files to use and hyperparameters
from settings import *

# functions that I have used, plus new functions to read files.

# adds data to a list for training or testing. Could add an 
# even amount or uneven amount based on inputs.
def add_to_data(data_set, tumor_max, normal_max, used_tumor, used_normal):
	normal_data = open(text_file_normal, "r")
	tumor_data = open(text_file_tumor, "r")
	if used_tumor > 0:
		for i in range(used_tumor):
			tumor_data.readline()
	if used_normal > 0:
		for i in range(used_normal):
			normal_data.readline()
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

def split_data(input_vector, gene_groups):
	input_vector = input_vector.tolist()[0]
	split = [] # should be of length of gene_groups
	gene_indexer = gene_dict()
	for gene_group in gene_groups:
		gene_group_indices = get_gene_indicies(gene_group, gene_indexer)
		subset = list(map(input_vector.__getitem__, gene_group_indices))
		split.append(subset)
	return split

# NN class 
class NN(nn.Module):
	def __init__(self, hidden_size, output_size):
		super(NN, self).__init__()
		# no longer care about input size, it will be variable
		# number of linear layers is based on hidden size
		fc = [] #should be length of hidden_size 
		
		# need gene group data lengths to make layers
		group_lengths = []
		self.gene_groups = import_gene_groups()
		gene_indexes = gene_dict()
		for gene_group in self.gene_groups:
			group_lengths.append(get_gene_indicies(gene_group, gene_indexes))
		for i in range(hidden_size):
			fc.append(nn.Linear(len(group_lengths[i]), 1))
		self.linears = nn.ModuleList(fc)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, 2)
	def forward(self, input_vector):
		# here we must split the input_vector and active it
		# using the specific group
		hidden = []
		input_split = split_data(input_vector, self.gene_groups)
		for i in range(hidden_size):
			x = torch.FloatTensor(input_split[i])
			hidden.append(self.linears[i](x))
		hidden = torch.stack(hidden, 1)
		out = self.relu(hidden)
		out= self.fc2(out)
		return out



if __name__ == "__main__":
	start_time = time.monotonic()

	### Hyperparameters
	hidden_size = len(import_gene_groups()) 
	
	# terminal message to track work
	print("Building the partial model trained on the", mode, tumor_data_size, "Tumor and", normal_data_size, "Normal samples.")
	print("Hyperparameters:", num_epochs, "epochs,", hidden_size, "neurons in the hidden layer,", learning_rate, "learning rate.")

	model = NN(hidden_size, output_size)
	model = model.train()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	## this constant loading and configuring
	training_data = []
	add_to_data(training_data, tumor_data_size, normal_data_size, 0, 0)

	# train the model
	print("Training the model")
	for epoch in range(num_epochs):
		print(epoch + 1, "out of", num_epochs, end="", flush=True)
		random.shuffle(training_data)
		for instance, label in training_data:
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
		sys.stdout.write("\r")
		sys.stdout.flush()
	
	trained_dict = copy.deepcopy(model.state_dict())
	trained = trained_dict[list(trained_dict.keys())[0]]

	print("Saving the model to file")
	torch.save(model.state_dict(),"state_dicts/nn_split.pt")
	end_time = time.monotonic()
	print("Runtime:", timedelta(seconds=end_time - start_time))
	print()
