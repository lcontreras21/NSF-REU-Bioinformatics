# Test the model without needing to retrain the model every time
# This allows for multiple tests without waiting for retraining
# Import model from saved file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
from nn_1 import NN, add_to_data, make_gene_vector, make_target, label_to_ix

start_time = time.time()

normal_data = open("normal.txt", "r")
tumor_data = open("tumor.txt", "r")
training_size = int(sys.argv[4])
for i in range(training_size):
	normal_data.readline()
	tumor_data.readline()

print("Importing the model from the saved location")
input_size, output_size, hidden_size = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
model = NN(output_size, hidden_size, input_size)
model.load_state_dict(torch.load("nn_1_state_dict.pt"))
model.eval()

testing_size = int(sys.argv[5])

current_trial, trials = 1, 1
if len(sys.argv) == 7:
	trials = int(sys.argv[6])
print("Testing accuracy of model with trial count of: ", trials)
for trial in range(trials):
	testing_data = []
	add_to_data(testing_data, testing_size, normal_data, tumor_data)
	
	print("Trial: ", trial + 1)
	with torch.no_grad():
		correct, total, index = 0, 0, 0
		increment = len(testing_data) // 10
		for instance, label in testing_data:
			if index % increment == 0:
				print("-" * (index // increment), end="")

			index += 1
			gene_vec = make_gene_vector(instance, input_size)
			target = make_target(label, label_to_ix)
			outputs = model(gene_vec)
			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()
			sys.stdout.write("\r")
			sys.stdout.flush()
		print("-" * 10)
		print("Correct: " + str(correct), "Total: "+ str(total))
print((time.time() - start_time) // 60)
normal_data.close()
tumor_data.close()
