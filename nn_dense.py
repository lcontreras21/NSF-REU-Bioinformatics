import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import time
from datetime import timedelta
import random
from settings import *
from tqdm import tqdm

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

class NN_dense(nn.Module):
	def __init__(self, gene_size, hidden_size, num_labels):
		super(NN_dense, self).__init__()
		self.fc1 = nn.Linear(gene_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_labels)
		#self.fc1 = nn.Linear(gene_size, num_labels)
	def forward(self, gene_vec):
		out = self.fc2(self.relu(self.fc1(gene_vec)))
		#out = self.fc1(gene_vec)
		out = F.log_softmax(out, dim=1)
		return out
	def __str__(self):
		return "Dense"

def make_gene_vector(file_line, input_size):
	data = list(map(float, file_line))
	vec = torch.FloatTensor(data)
	return vec.view(1, -1)

def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])

# convert tumor or no tumor to int value
label_to_ix = {"Tumor": 0, "Normal": 1}

def train_dense_model():
	start_time = time.monotonic()

	# hyper parameters
	hidden_size = 50

	# text files for normal and tumor data
	normal_data = open(text_file_normal, "r")
	tumor_data = open(text_file_tumor, "r")
	
	if debug:
		print("Building the dense model trained on the", mode,
			tumor_data_size, "Tumor and", 
			normal_data_size, "Normal samples.")
		print("Hyperparameters:", 
			num_epochs, "epochs,", 
			hidden_size, "neurons in the hidden layer,", 
			learning_rate, "learning rate.")
	
	# building the model and loading other functions
	model = NN_dense(input_size, hidden_size, output_size)
	model = model.train()
	loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	if debug:
		print("Loading the data", end='', flush=True)	
	training_data = []
	add_to_data_uneven(training_data, tumor_data_size, normal_data_size,  normal_data, tumor_data)

	normal_data.close()
	tumor_data.close()
	sys.stdout.write("\r")
	sys.stdout.flush()
	if debug:
		print("Loaded the data ", flush=True)
		print("Training the model")
	for epoch in tqdm(range(num_epochs), disable=not debug):
		random.shuffle(training_data)
		for i in tqdm(range(len(training_data)), disable=not debug):
			instance, label = training_data[i]
			
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
	if debug:
		print()
		print("Saving the model to file")
	torch.save(model.state_dict(), stored_dict_locs[str(model)])
	if debug:
		print("Time elapsed: ", timedelta(seconds=time.monotonic() - start_time))
		print()
	
if __name__ == "__main__":
	train_dense_model()
