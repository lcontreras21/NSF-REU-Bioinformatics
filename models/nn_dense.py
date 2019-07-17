# Simple neural network, all the nodes are connected.
import torch.nn as nn
from settings import *
from models.process_data import *

class NN_dense(nn.Module):
	def __init__(self, gene_size, hidden_size, num_labels):
		super(NN_dense, self).__init__()
		self.fc1 = nn.Linear(gene_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_labels)
		self.mask()

	def forward(self, input_vector):
		out =  self.fc1(input_vector)
		out = self.relu(out)
		out = self.fc2(out)
		out = F.log_softmax(out, dim=1)
		
		self.mask()
		return out
	def __str__(self):
		return "Dense"

	def mask(self):
		# bias mask
		mask = np.array([1] * hidden_size)
		mask[weights_to_test] = 0
		self.fc1.bias.data *= torch.FloatTensor(mask)

		# weight mask
		mask = np.array([[1] * input_size] * hidden_size)
		mask[weights_to_test] = 0
		self.fc1.weight.data *= torch.FloatTensor(mask)

def train_dense_model():
	train_model(NN_dense(input_size, hidden_size, output_size))

if __name__ == "__main__":
	train_dense_model()
