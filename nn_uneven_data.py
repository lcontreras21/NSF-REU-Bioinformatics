# neural network attempt number 4 
# creating a basic neural network that's stock from python

# using tutorial found at https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#creating-network-components-in-pytorch

# example of logarithmic and nonlinear implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import random

def add_to_data_uneven(data_set, tumor_max, normal_max, normal_data, tumor_data):
	have_n_tumors = 0
	have_n_normals = 0
	# there should be equal amount of tumor and normal data
	while have_n_tumors < tumor_max:
		next_line = tumor_data.readline().split()
		data_set += [(next_line[1:-1], next_line[-1])]
		have_n_tumors += 1
	while have_n_normals < normal_max :
		next_line = normal_data.readline().split()
		data_set += [(next_line[1:-1], next_line[-1])]
		have_n_normals += 1

class NN(nn.Module):
	def __init__(self, num_labels, hidden_size, gene_size):
		super(NN, self).__init__()
		self.fc1 = nn.Linear(gene_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_labels)
		#self.fc1 = nn.Linear(gene_size, num_labels)	
	def forward(self, gene_vec):
		out = self.fc1(gene_vec)
		out = self.relu(out)
		out = self.fc2(out)
		#out = F.log_softmax(out, dim=1)
		return out
def make_gene_vector(file_line, input_size):
	data = list(map(float, file_line))
	vec = torch.FloatTensor(data)
	return vec.view(1, -1)

def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])

# convert tumor or no tumor to int value
label_to_ix = {"Tumor": 0, "Normal": 1}

if __name__ == "__main__":
	start_time = time.time()
	# read in the file for gene
	#f = open("..\subset_0.1_logged_scaled_rnaseq_hk_normalized.txt", "r")
	#gene_info = f.readline().split()
	
	# convert gene location to integer value
	#gene_to_ix = {}
	#for gene_name in gene_info[1:-1]:
	#	if gene_name not in gene_to_ix:
	#		gene_to_ix[gene_name] = len(gene_to_ix)
	#f.close()
	

	# hyper parameters
	input_size = 35728 
	output_size = 2
	num_epochs = 3 
	hidden_size = 50
	learning_rate = 0.01

	# text files for normal and tumor data
	normal_data = open("text_files/normal.txt", "r")
	tumor_data = open("text_files/tumor.txt", "r")
	print("Loading the data")
	
	training_data = []
	tumor_data_size = int(sys.argv[1])
	normal_data_size = int(sys.argv[2])
	
	add_to_data_uneven(training_data, tumor_data_size, normal_data_size,  normal_data, tumor_data)

	normal_data.close()
	tumor_data.close()

	print("Building the model trained on", tumor_data_size, "-", normal_data_size, "input, with", num_epochs, "epochs and", hidden_size, "neurons in the hidden layer.")
	print("Learning rate is", learning_rate)

	# building the model and loading other functions
	model = NN(output_size, hidden_size, input_size)
	model = model.train()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	print("Training the model")
	for epoch in range(num_epochs):
		print(epoch + 1, "out of", num_epochs, end="")
		random.shuffle(training_data)
		for instance, label in training_data:
			# erasing gradients from previous run
			model.zero_grad()
			
			gene_vec = make_gene_vector(instance, input_size)
			target = make_target(label, label_to_ix)
			
			# getting probabilities from instance
			output = model(gene_vec)
				
			# applying learning to the model
			loss = loss_function(output, target)
			loss.backward()
			optimizer.step()
		# progress tracker
		sys.stdout.write("\r")
		sys.stdout.flush()
	print("Saving the model to file")
	torch.save(model.state_dict(), "state_dicts/nn_base.pt")
	print("Time elapsed: ", (time.time() - start_time) // 60)
