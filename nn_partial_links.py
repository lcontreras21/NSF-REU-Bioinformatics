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
	f = open("text_files\\logged_scaled_rnaseq.txt", "r")
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

# using hallmark gene groups, set the weights in the first layer to zero
# for the genes that are not in the hallmark group.
# for example, first node in hidden layer is first hallmark group and all
# genes not in that group will have value of zero in the input
def set_weights(model_state): 
	gene_groups = import_gene_groups()
	gene_info = gene_dict()

	# we are interested in the first layer in model_state
	layer = model_state[list(model_state)[0]]
	# this returns a tensor of dimension (size of gene_group, 20629)
	
	# based on gene_groups data, set value equal to zero if not in group
	# iterate through gene groups
	# iterate through each gene in gene group
	# get its ix from gene_dict
	# set the value of it equal to zero based on gene group index and ix
	
	group_count = 0
	for group in gene_groups:
		for gene in group:
			try:
				gene_index = gene_info[gene]
				layer[group_count, gene_index] = 0
			except:
				pass
		group_count += 1

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
	input_size = 20629
	output_size = 2
	num_epochs = 3
	hidden_size = len(import_gene_groups()) 
	learning_rate = 0.01
	
	# terminal message to track work
	print("Building model trained on", sys.argv[1], "Tumor and", sys.argv[2], "Normal samples.")
	print("Hyperparameters:", num_epochs, "epochs,", hidden_size, "neurons in hidden layer,", learning_rate, "learning rate")

	model = NN(input_size, hidden_size, output_size)
	model = model.train()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	# get data size from the terminal and make practice training set
	## todo: make separate files for training and testing to reduce
	## this constant loading and configuring
	tumor_max = int(sys.argv[1])
	normal_max = int(sys.argv[2])
	training_data = []
	add_to_data(training_data, tumor_max, normal_max)


	
	# set the starting weights to model the biology
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
			optimizer.step()
			set_weights(model.state_dict())
		sys.stdout.write("\r")
		sys.stdout.flush()
	
	trained_dict = copy.deepcopy(model.state_dict())
	trained = trained_dict[list(trained_dict.keys())[0]]
	
	print("Checking if model weights don't change")
	print("Before:",untrained[49, 6298])
	print("After:", trained[49, 6298])
	print("Before:",untrained[49, 7076])
	print("After:", trained[49, 7076])

	print("Saving the model to file")
	torch.save(model.state_dict(),"state_dicts\\nn_partial_links.pt")
	end_time = time.monotonic()
	print("Runtime:", timedelta(seconds=end_time - start_time))
