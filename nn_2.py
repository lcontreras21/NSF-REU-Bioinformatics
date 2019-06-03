# Second attempt at creating a neural network
# This time, it will be linear model following 
# the example at https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import sys
import time

start_time = time.process_time()
# read in the text file
f = open("..\subset_0.1_logged_scaled_rnaseq_hk_normalized.txt", "r")
gene_info = f.readline().split()

training_data = []
testing_data = []
testing_loader = []
def add_to_data(data_set, data_max):
	have_n_tumors = 0
	have_n_normals = 0
	while len(data_set) < data_max * 2:
		next_line = f.readline().split()
		if next_line[-1] == "Tumor" and have_n_tumors < data_max:
			data_set += [(next_line[1:-1], next_line[-1])]
			have_n_tumors += 1
		elif next_line[-1] == "Normal" and have_n_normals < data_max:
			data_set += [(next_line[1:-1], next_line[-1])]
			have_n_normals += 1
print("Loading in the data")
add_to_data(training_data, 30)
add_to_data(testing_data, 10)
add_to_data(testing_loader, 5)


f.close()

def check_data(data_set):
	for instance, label in data_set:
		print(label)

def make_gene_vector(file_line):
	vec = torch.zeros(input_size)
	index = 0
	for gene in file_line:
		vec[index] += float(gene)
		index += 1
	return vec.view(1, -1)

# convert tumor or no tumor to int value
label_to_ix = {"Tumor": 0, "Normal": 1}

def make_target(label, label_to_ix):
	return torch.LongTensor([label_to_ix[label]])

def test_model():
	with torch.no_grad():
		for instance, label in testing_data:
			gene_vec = make_gene_vector(instance)
			outputs = model(gene_vec)
			print(outputs, label)

# hyper parameters
input_size = len(gene_info[1:-1])
hidden_size = 6500 
num_classes = 2
num_epochs = 3 
learning_rate = 0.001

print("Creating the model")
class NN(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(NN, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_classes)
	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out

model = NN(input_size, hidden_size, num_classes)

#print("Testing the model before training")
#test_model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training the model")
total_step = len(training_data)
for epoch in range(num_epochs):
	print(epoch + 1, "out of", num_epochs, end="")
	for instance, label in training_data:
		instance = make_gene_vector(instance)

		outputs = model(instance)
		target = make_target(label, label_to_ix)
		loss = criterion(outputs, target)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step
	sys.stdout.write('\r')
	sys.stdout.flush()
#print("Testing the model after training")
#test_model()

print("Checking the accuracy of the model")
with torch.no_grad():
	correct = 0
	total = 0
	for instance, label in testing_loader:
		gene_vec = make_gene_vector(instance)
		target = make_target(label, label_to_ix)
		outputs = model(gene_vec)
		_, predicted = torch.max(outputs.data, 1)
		total += target.size(0)
		correct += (predicted == target).sum().item()
	print(correct, total)


print(time.process_time() - start_time)
