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


# functions that I have used, plus new functions to read files.

# adds data to a list for training or testing. Could add an 
# even amount or uneven amount based on inputs.
def add_to_data(data_set, tumor_max, normal_max):
	normal_data = open("text_files\\normal.txt", "r")
	tumor_data = open("text_files\\tumor.txt", "r")
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
		tumors += 1
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
	f = open("..\subset_0.1_logged_scaled_rnaseq_hk_normalized.txt", "r")
	gene_names = f.readline().split()
	f.close()

	gene_to_index = {}
	for gene_name in gene_names[1:-1]:
		if gene_name not in gene_to_index:
			gene_to_index[gene_name] = len(gene_to_index)
	return gene_to_index

# import hallmark gene data
# should probably make this dynamic for other files
def import_gene_groups():
	f = open("text_files\\h.all.v6.2.symbols.txt", "r")
	gene_groups = []
	for line in f:
		gene_data = line.split()
		# returning list of lists
		gene_groups.append(gene_data[2:])
	f.close()
	return gene_groups


if __name__ == "__main__":
	start_time = time.monotonic()
	
	### Hyperparameters
	input_size = 20629
	output_size = 2
	num_epochs = 3
	hidden_size = # make this variable based on gene group data
	learning_rate = 0.01
	
	# terminal message to track work
	print("Building model trained on data of Tumor:", tumor_samples, "Normal:", normal_samples)
	print("Hyperparameters:", num_epochs, "epochs,", hidden_size, "neurons in hidden layer,", learning_rate, "learning rate")


	end_time = time.monotonic()
	print("Runtime:", timedelta(seconds=end_time - start_time))
