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

from nn_partial_links import add_to_data, make_gene_vector, make_target, label_to_ix, NN
from nn_split import NN_split
from nn_dense import NN_dense
from settings import *

def test_model(model, testing_data):
	if debug:
		print("Testing accuracy of", model, "model using the", data, "dataset")
	# keeping a count to test the accuracy of model
	with torch.no_grad():
		correct, total, true_pos, true_neg, total_pos, total_neg = 0, 0, 0, 0, 0, 0
		for i in tqdm(range(len(testing_data)), disable=not debug):
			instance, label = testing_data[i]
			
			gene_vec = make_gene_vector(instance)
			target = make_target(label, label_to_ix)
			outputs = model(gene_vec)
			_, predicted = torch.max(outputs.data, 1)
			total += target.size(0)
			correct += (predicted == target).item()
			
			if target.item() == 1:  # getting normal is considered True, tumor is False
				total_pos += 1
				if torch.equal(predicted, target):
					true_pos += 1  # true positive if predicted == actual
			elif target.item() == 0:
				total_neg += 1
				if torch.equal(predicted, target):
					true_neg += 1  # true negative if predicted == actual
			
	# account for specificity and sensitivity here
	if debug:
		print("Sensitivity:   ", "{:04}".format(true_pos), "/", "{:04}".format(total_pos), "=", true_pos/total_pos) # sensitivity = true positive / total positive
		print("Specificity:   ", true_neg, "/", total_neg, "=", true_neg/total_neg) # specificity = true negative / total negative
		percent_correct = correct / total
		print("Total Correct: ", correct, "/", total, "=", percent_correct)
	if record_data: 
		if debug:
			print("Recording data")
		f = open(percent_save_loc, "a")
		f.write(model.__str__() + "\t" + str(true_pos/total_pos) + "\t" + str(true_neg/total_neg) + "\t" + str(correct / total) + "\n")
		f.close()

def test_models():
	start_time = time.monotonic()

	# Setting text files and other stuff to use
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

	model_partial = NN(input_size, hidden_size, output_size)
	model_partial.load_state_dict(torch.load(partial_dict))

	model_dense = NN_dense(input_size, hidden_size, output_size)
	model_dense.load_state_dict(torch.load(dense_dict))

	model_split = NN_split(hidden_size, output_size)
	model_split.load_state_dict(torch.load(split_dict))

	model_dense.eval()
	model_partial.eval()
	model_split.eval()

	testing_data = []
	add_to_data(testing_data, tumor_data_size, normal_data_size)
	random.shuffle(testing_data)

	if debug:
		print()
		print("Zero-weights model")
	test_model(model_partial, testing_data)
	if debug:
		print()

		print("Dense model")
	test_model(model_dense, testing_data)
	if debug:
		print()

		print("Split model")
	test_model(model_split, testing_data)
	if debug:
		print()

		end_time = time.monotonic()
		print("Runtime:", timedelta(seconds=end_time - start_time))

if __name__ == "__main__":
	test_models()	
