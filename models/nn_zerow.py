# neural network where the hidden layer is connected 
# to the input based on gene groups. ie not all nodes
# are connected
import torch.nn as nn
from models.process_data import *
# settings file that has what files to use and hyperparameters
from settings import *
from train_models import *
import numpy as np

class NN_zerow(nn.Module):
	def __init__(self):
		super(NN_zerow, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, output_size)
		
		self.load_starting_seed()
		self.make_mask()
		self.mask()
	
	def forward(self, input_vector):
		out = self.fc1(input_vector)
		out = self.relu(out)
		out = self.fc2(out)
		out = torch.sigmoid(out)

		return out

	def make_mask(self):
		gene_group_indicies = read_indicies()
		mask = np.array([[0] * input_size] * len(gene_group_indicies))
		for i, gene_group in enumerate(gene_group_indicies):
			if test_behavior and i in weights_to_test:
				gene_group = []
			mask[i][gene_group] = 1
		mask = torch.FloatTensor(mask)
		self.weight_mask = mask

		mask = np.array([1] * hidden_size)
		mask[weights_to_test] = 0
		self.bias_mask = torch.FloatTensor(mask)

	def load_starting_seed(self):
		self.fc1.weight.data, self.fc2.weight.data = get_starting_seed()

	def mask(self):
		self.fc1.weight.data *= self.weight_mask
		self.fc1.bias.data *= self.bias_mask

	def __str__(self):
		return "Zero-weights"

if __name__ == "__main__":
	training_data = load_training_data()
	train_model(NN_zerow(), training_data)
