# neural network attempt number 1
# creating a basic neural network that's stock from python

# using tutorial found at https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#creating-network-components-in-pytorch

# example of logarithmic and nonlinear implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# read in the file for gene
f = open("..\subset_0.1_logged_scaled_rnaseq_hk_normalized.txt", "r")
gene_info = f.readline().split()

# convert gene location to integer value
gene_to_ix = {}
for gene_name in gene_info[1:-1]:
	if gene_name not in gene_to_ix:
		gene_to_ix[gene_name] = len(gene_to_ix)

training_data = []
test_data = []
def add_to_data(data_set, data_max):
	have_n_tumors = 0
	have_n_normals = 0
	while len(data_set) < 10:
		next_line = f.readline().split()
		if next_line[-1] == "Tumor" and have_n_tumors < data_max:
			data_set += [(next_line[1:-1], next_line[-1])]
			have_n_tumors += 1
		elif next_line[-1] == "Normal" and have_n_normals < data_max:
			data_set += [(next_line[1:-1], next_line[-1])]
			have_n_normals += 1

add_to_data(training_data, 5)
add_to_data(test_data, 5)

f.close()

class NN(nn.Module):
	def __init__(self, num_labels, gene_size):
		super(NN, self).__init__()
		self.linear = nn.Linear(gene_size, num_labels)
		# num_labels will be 2 indicating if tumor or not

	def forward(self, gene_vec):
		#return F.log_softmax(self.linear(gene_vec), dim=1)
		return self.linear(gene_vec)
def make_gene_vector(file_line):
	vec = torch.zeros(20629)
	index = 0
	for gene in file_line:
		vec[index] += float(gene)
		index += 1
	return vec.view(1, -1)

def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])

model = NN(2, 20629)

# display parameters
#for param in model.parameters():
#	print(param)

with torch.no_grad():
	for instance, label in test_data:
		gene_vector = make_gene_vector(instance)
		log_probs = model(gene_vector)
		print(log_probs, label)

# convert tumor or no tumor to int value
label_to_ix = {"Tumor": 0, "Normal": 1}

# print the column corresponding to TSPAN6
print(next(model.parameters())[:, gene_to_ix["TSPAN6"]])

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(5):
	for instance, label in training_data:
		model.zero_grad()
		gene_vec = make_gene_vector(instance)
		target = make_target(label, label_to_ix)
		log_probs = model(gene_vec)
		
		loss = loss_function(log_probs, target)
		loss.backward()
		optimizer.step()

with torch.no_grad():
	for instance, label in test_data:
		gene_vec = make_gene_vector(instance)
		log_probs = model(gene_vec)
		print(log_probs, label)

print(next(model.parameters())[:, gene_to_ix["TSPAN6"]])
