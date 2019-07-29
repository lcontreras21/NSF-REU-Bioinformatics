'''
 neural network where the structure of the network is 
 explicitely defined and the weights are not made zero
'''
import torch.nn as nn
from models.process_data import *
from train_models import *
from settings import *
from operator import itemgetter
import numpy as np

class NN_split(nn.Module):
	def __init__(self):
		super(NN_split, self).__init__()
		self.linears = nn.ModuleList(self.make_linears())
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size - len(weights_to_test), output_size)

		self.load_starting_seed()

	def forward(self, input_vector):
		out = self.fc1(input_vector) #custom fc1 compared to other models
		out = self.relu(out)
		out = self.fc2(out)
		#out = F.log_softmax(out, dim=1)
		out = torch.sigmoid(out)
		return out

	def fc1(self, input_vector):
		# split input, active using it's specific linear function 
		data = self.split_data(input_vector)
		hidden = [self.linears[i](data[i]) for i in range(hidden_size) if i not in weights_to_test]
		# concatenate all the linear layers to make a tensor with size of gene groups
		hidden = torch.stack(hidden, 1)
		return hidden
	
	def make_linears(self):
		self.gene_group_indicies, fc = [], []
		self.gene_group_indicies = read_indicies()
		for i, gene_group in enumerate(gene_group_indicies):
			if test_behavior and i in weights_to_test:
				fc.append(nn.Linear(1,1))
				fc[-1].weight = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)
				fc[-1].bias = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)
				self.gene_group_indicies.append([])
			else:
				fc.append(nn.Linear(len(gene_group), 1))
		return fc

	def split_data(self, input_vector):
		data = input_vector.tolist()[0]
		split = []
		for group_indices in self.gene_group_indicies:
			try:
				trimmed_data = itemgetter(*group_indices)(data)
			except:
				trimmed_data = []
			split.append(torch.FloatTensor(trimmed_data))
		return split

	def load_starting_seed(self):
		seed = get_starting_seed()
		split_seed = [i.unsqueeze(0) for i in self.split_data(seed)]
		for i, layer in enumerate(self.linears):
			self.linears[i].weight.data = split_seed[i]


	def __str__(self):
		return "Split"

if __name__ == "__main__":
	training_data = load_training_data()
	train_model(NN_split(), training_data)
