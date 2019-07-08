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
		text_file_normal = test_dir + "full_test_normal_samples.txt" 
		text_file_tumor = test_dir + "full_test_tumor_samples.txt"
		text_data = logged
	elif data == "subset":
		text_file_normal = test_dir + "subset_test_normal_samples.txt"
		text_file_tumor = test_dir + "subset_test_tumor_samples.txt"
		text_data = sub_logged
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

used_tumor = tumor_data_size 
used_normal = normal_data_size

testing_data = []
# test with even data to see results
from nn_partial_links import add_to_data, make_gene_vector, make_target, label_to_ix 

if mode[0] != "t":
	add_to_data(testing_data, samples_per_trial, samples_per_trial)
else:
	add_to_data(testing_data, tumor_data_size, normal_data_size)
used_tumor += samples_per_trial
used_normal += samples_per_trial
random.shuffle(testing_data)

def test_model(model):
	if debug:
		print("Testing accuracy of", model, "using", data, "dataset")
	# keeping a count to test the accuracy of model
	with torch.no_grad():
		correct, total, =  0, 0
		true_pos, true_neg = 0, 0
		total_pos, total_neg = 0, 0
		for i in tqdm(range(len(testing_data)), disable=not debug):
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
		#print("Sensitivity:", true_pos, "/", total_pos, "=", true_pos/total_pos)
		#print("Specificity:", true_neg, "/", total_neg, "=", true_neg/total_neg)
		
		if debug:
			print(true_pos/total_pos)
			print(true_neg/total_neg)
		
	percent_correct = correct / total
	#print("Total Correct: ", total_correct, "/", Total, "=", percent_correct)
	if debug:
		print(percent_correct)

	if record_data: 
		if debug:
			print("Recording data")
		f = open(percent_save_loc, "a")
		f.write(model.__str__() + "\t" + str(true_pos/total_pos) + "\t" + str(true_neg/total_neg) + "\t" + str(percent_correct) + "\n")
		f.close()

if debug:
	print()
	print("Zero-weights model")
test_model(model_partial)
if debug:
	print()

	print("Dense model")
test_model(model_dense)
if debug:
	print()

	print("Split model")
test_model(model_split)
if debug:
	print()

	end_time = time.monotonic()
	print("Runtime:", timedelta(seconds=end_time - start_time))
