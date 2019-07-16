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
	def forward(self, gene_vec):
		out = self.fc2(self.relu(self.fc1(gene_vec)))
		out = F.log_softmax(out, dim=1)
		return out
	def __str__(self):
		return "Dense"

def train_dense_model():
	train_model(NN_dense(input_size, hidden_size, output_size))

if __name__ == "__main__":
	train_dense_model()
