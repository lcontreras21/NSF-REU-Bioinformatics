# neural network where the hidden layer is connected 
# to the input based on gene groups. ie not all nodes
# are connected

# this will be the beginning of the full project 
# probably not the final file but a lot of original code
# will go here for simplicity and to avoid copying

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
	#f = open("\subset_0.1_logged_scaled_rnaseq_hk_normalized.txt", "r")
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
		# returning list of lists
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

# using hallmark gene groups, set the weights in the first layer to zero
# for the genes that are not in the hallmark group.
# for example, first node in hidden layer is first hallmark group and all
# genes not in that group will have value of zero in the input
def set_weights(model_state): 
	gene_groups = import_gene_groups()
	gene_indexer = gene_dict()

	# we are interested in the first layer in model_state
	layer = model_state[list(model_state)[0]]
	# this returns a tensor of dimension (size of gene_group, size of input)
	
	# go through each group, for hallmark there are 50 groups
	total = 0
	for group_index, gene_group in enumerate(gene_groups):
		# then go through each gene in that group
		# genes not in that group should have weight of 0
	
		# get current row and use it as a list
		layer_list = layer[group_index].tolist()
	
		# list containing which indices to keep
		gene_group_indices = get_gene_indicies(gene_group, gene_indexer) 
		gene_group_indices += [len(layer_list)]
		previous_gene = 0

		# using python list manipulation to quickly set which genes to zero based on the gene indices.
		for gene in gene_group_indices:
			layer_list[previous_gene: gene] = [0.0] * (gene - previous_gene)
			previous_gene = gene + 1
		layer[group_index] = torch.LongTensor(layer_list)
		

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
		return out


if __name__ == "__main__":
	start_time = time.monotonic()

	### Hyperparameters
	hidden_size = len(import_gene_groups()) 
	
	# terminal message to track work
	print("Building the partial model trained on the", mode, tumor_data_size, "Tumor and", normal_data_size, "Normal samples.")
	print("Hyperparameters:", num_epochs, "epochs,", hidden_size, "neurons in the hidden layer,", learning_rate, "learning rate.")

	model = NN(input_size, hidden_size, output_size)
	model = model.train()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# get data size from the terminal and make practice training set
	## todo: make separate files for training and testing to reduce
	## this constant loading and configuring
	training_data = []
	add_to_data(training_data, tumor_data_size, normal_data_size, 0, 0)


	
	# set the starting weights to model the biology
	print("Setting the weights of the model")
	set_weights(model.state_dict())

	# making a copy to see if changes are permanent
	untrained_dict = copy.deepcopy(model.state_dict())
	untrained = untrained_dict[list(untrained_dict.keys())[0]]
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
			set_weights(model.state_dict())
			optimizer.step()
		sys.stdout.write("\r")
		sys.stdout.flush()
	
	trained_dict = copy.deepcopy(model.state_dict())
	trained = trained_dict[list(trained_dict.keys())[0]]

	print("Checking to see weighs are still zero")
	print("Untrained | Trained")
	print(untrained[8,3941], trained[8,3941])

	print("Saving the model to file")
	torch.save(model.state_dict(),"state_dicts/nn_partial_links.pt")
	end_time = time.monotonic()
	print("Runtime:", timedelta(seconds=end_time - start_time))
	print()
