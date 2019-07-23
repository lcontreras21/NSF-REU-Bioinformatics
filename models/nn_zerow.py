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
		#self.set_weights()
	
	def forward(self, input_vector):
		out = self.fc1(input_vector)
		out = self.relu(out)
		out = self.fc2(out)
		out = F.log_softmax(out, dim=1)

		self.set_bias()
		self.set_weights()
		return out

	def make_mask(self):
		gene_groups = import_gene_groups()
		gene_indexer = gene_dict()
		mask = np.array([[0] * input_size] * len(gene_groups))
		for i, gene_group in enumerate(gene_groups):
			if test_behavior and i in weights_to_test:
				group_indices = []
			else:
				group_indices = get_gene_indicies(gene_group, gene_indexer)
			mask[i][group_indices] = 1
		mask = torch.FloatTensor(mask)
		self.mask = mask

	def load_starting_seed(self):
		self.fc1.weight.data = get_starting_seed()

	def set_weights(self):
		self.fc1.weight.data *= self.mask

	def set_bias(self):
		mask = np.array([1] * hidden_size)
		mask[weights_to_test] = 0
		self.fc1.bias.data *= torch.FloatTensor(mask)

	def __str__(self):
		return "Zero-weights"

def train_zerow_model(training_data):
	train_model(NN_zerow(), training_data)

if __name__ == "__main__":
	training_data = load_training_data()
	train_zerow_model(training_data)
