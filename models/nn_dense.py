# Simple neural network, all the nodes are connected.
import torch.nn as nn
from settings import *
from models.process_data import *
from train_models import *
import numpy as np
from copy import deepcopy

class NN_dense(nn.Module):
	def __init__(self, hidden_size=hidden_size):
		super(NN_dense, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, output_size)
		
		self.hidden_size = hidden_size
		self.load_starting_seed()
		self.make_mask()
		self.mask()

	def forward(self, input_vector):
		out = self.fc1(input_vector)
		out = self.relu(out)
		out = self.fc2(out)
		out = torch.sigmoid(out)

		self.mask()
		return out
	
	def __str__(self):
		return "Dense"
	
	def load_starting_seed(self):
		self.fc1.weight.data, self.fc2.weight.data = get_starting_seed()

	def mask(self):
		self.fc1.bias.data *= self.bias_mask
		self.fc1.weight.data *= self.weight_mask

	def make_mask(self):
		# bias mask
		mask = np.array([1] * self.hidden_size)
		mask[weights_to_test] = 0
		self.bias_mask = torch.FloatTensor(mask)

		# weight mask
		mask = np.array([[1] * input_size] * self.hidden_size)
		mask[weights_to_test] = 0
		self.weight_mask = torch.FloatTensor(mask)

if __name__ == "__main__":
	training_data = load_training_data()
	train_model(NN_dense(), training_data)
