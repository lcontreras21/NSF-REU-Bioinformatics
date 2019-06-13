# Test the model without needing to retrain the model every time
# This allows for multiple tests without waiting for retraining
# Import model from saved file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
from nn_partial_links import * 
import random
from datetime import timedelta

start_time = time.monotonic()

print("Importing the model from the saved location")
input_size, output_size, hidden_size = 35728, 2, int(sys.argv[1])
model_partial = NN(input_size, hidden_size, output_size)
model_partial.load_state_dict(torch.load("state_dicts/nn_partial_links.pt"))
model_partial.eval()

model_base = NN(input_size, hidden_size, output_size)
model_base.load_state_dict(torch.load("state_dicts/nn_base.pt"))
model_base.eval()

tumor_data_size = int(sys.argv[2])
normal_data_size = int(sys.argv[3])
samples_per_trial = int(sys.argv[4])


current_trial, trials = 1, 1

# if we are given trials as input, use that, otherwise use default 1
if len(sys.argv) == 6:
	trials = int(sys.argv[5])
print("Testing accuracy of model with trial count of: ", trials)
def test_model(model):
	# keeping a count to test the accuracy of model
	total_correct = 0
	total_from_trials = 0
	trials_pos, trials_neg = 0, 0
	used_tumor = tumor_data_size
	used_normal = normal_data_size
	for trial in range(trials):	
		testing_data = []
		# test with even data to see results
		add_to_data(testing_data, samples_per_trial, samples_per_trial, used_tumor, used_normal)
		used_tumor += samples_per_trial
		used_normal += samples_per_trial
		random.shuffle(testing_data)
		print("Trial: ", trial + 1)
		with torch.no_grad():
			correct, total, index = 0, 0, 0
			true_pos, true_neg = 0, 0
			total_pos, total_neg = 0, 0
			increment = len(testing_data) // 10
			if increment == 0:
				increment = 10
			for instance, label in testing_data:
				if index % increment == 0:
					print("-" * (index // increment), end="")
				index += 1
				
				gene_vec = make_gene_vector(instance)
				target = make_target(label, label_to_ix)
				outputs = model(gene_vec)
				_, predicted = torch.max(outputs.data, 1)
				total += target.size(0)
				correct += (predicted == target).item()
				
				if target.item() == 1:
					total_pos += 1
					if torch.equal(predicted, target):
						true_pos += 1
				elif target.item() == 0:
					total_neg += 1
					if torch.equal(predicted, target):
						true_neg += 1
				
				sys.stdout.write("\r")
				sys.stdout.flush()	
		
			# getting normal is considered True, tumor is False

			# account for specificity and sensitivity here
			# sensitivity = true positive / total positive
			# specificity = true negative / total negative
			# true positive if predicted == actual
			# true negative if predicted == actual
			trials_pos += true_pos
			trials_neg += true_neg
			print("Sensitivity:", true_pos, "/", total_pos, "=", true_pos/total_pos)
			print("Specificity:", true_neg, "/", total_neg, "=", true_neg/total_neg)
			total_correct += correct
			total_from_trials += total
	percent_correct = total_correct / total_from_trials
	print("Total Correct: ", total_correct, "/", total_from_trials, "=", percent_correct)
	print("Overall Sensitivity: ", trials_pos, "/", total_from_trials / 2, "=", trials_pos / (total_from_trials / 2))
	print("Overall Specificity: ", trials_neg, "/", total_from_trials / 2, "=", trials_neg / (total_from_trials / 2))
print("Partial model")
test_model(model_partial)
print()
print("Base model")
test_model(model_base)
end_time = time.monotonic()
print("Runtime:", timedelta(seconds=end_time - start_time))
