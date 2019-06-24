# Test the model without needing to retrain the model every time
# This allows for multiple tests without waiting for retraining
# Import model from saved file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import random
from datetime import timedelta
from tqdm import tqdm

from settings import *

start_time = time.monotonic()

# Setting text files and other stuff to use
if mode != "free":
	if data == "full":
		text_file_normal = train_dir + "full_train_normal_samples.txt" 
		text_file_tumor = train_dir + "full_train_tumor_samples.txt"
		text_data = logged
	elif data == "subset":
		text_file_normal = train_dir + "subset_train_normal_samples.txt"
		text_file_tumor = train_dir + "subset_train_tumor_samples.txt"
		text_data = subset_logged
		input_size = 4579
	
	tumor_data_size = 1994
	normal_data_size = 146

input_size, output_size, hidden_size = input_size, 2, hidden_size
model_partial = NN(input_size, hidden_size, output_size)
model_partial.load_state_dict(torch.load(partial_dict))

model_dense = NN_dense(input_size, hidden_size, output_size)
model_dense.load_state_dict(torch.load(dense_dict))

model_split = NN_split(hidden_size, output_size)
model_split.load_state_dict(torch.load(split_dict))

model_dense.eval()
model_partial.eval()
model_split.eval()

current_trial = 1

print("Testing accuracy of model using", data, "with trial count of: ", trials)
def test_model(model):
	# keeping a count to test the accuracy of model
	total_correct, total_from_trials, trials_pos, trials_neg= 0, 0, 0, 0
	used_tumor = tumor_data_size 
	used_normal = normal_data_size
	for trial in range(trials):	
		testing_data = []
		# test with even data to see results
		if mode[0] != "t":
			add_to_data(testing_data, samples_per_trial, samples_per_trial, used_tumor, used_normal)
		else:
			add_to_data(testing_data, tumor_data_size, normal_data_size, 0, 0)
		used_tumor += samples_per_trial
		used_normal += samples_per_trial
		random.shuffle(testing_data)
		print("Trial: ", trial + 1)
		with torch.no_grad():
			correct, total, =  0, 0
			true_pos, true_neg = 0, 0
			total_pos, total_neg = 0, 0
			for i in tqdm(range(len(testing_data))):
				instance, label = testing_data[i]
				
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
				
			# getting normal is considered True, tumor is False

			# account for specificity and sensitivity here
			# sensitivity = true positive / total positive
			# specificity = true negative / total negative
			# true positive if predicted == actual
			# true negative if predicted == actual
			trials_pos += true_pos
			trials_neg += true_neg
			#print("Sensitivity:", true_pos, "/", total_pos, "=", true_pos/total_pos)
			#print("Specificity:", true_neg, "/", total_neg, "=", true_neg/total_neg)
			
			print(true_pos/total_pos)
			print(true_neg/total_neg)
			
			total_correct += correct
			total_from_trials += total
	percent_correct = total_correct / total_from_trials
	#print("Total Correct: ", total_correct, "/", total_from_trials, "=", percent_correct)
	print(percent_correct)
	if mode != "free":
		pass
	else:
		print("Overall Sensitivity: ", trials_pos, "/", total_from_trials / 2, "=", trials_pos / (total_from_trials / 2))
		print("Overall Specificity: ", trials_neg, "/", total_from_trials / 2, "=", trials_neg / (total_from_trials / 2))

print()
#print("Zero-weights model")
#test_model(model_partial)
#print()

print("Dense model")
test_model(model_dense)
print()

#print("Split model")
#test_model(model_split)
#print()

end_time = time.monotonic()
print("Runtime:", timedelta(seconds=end_time - start_time))
