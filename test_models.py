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

from models.process_data import * 
from models.nn_zerow import NN_zerow
from models.nn_split import NN_split
from models.nn_dense import NN_dense
from settings import *

def test_model(model, testing_data):
	if debug: 
		print("Testing accuracy of", model, "model using the", data, "dataset")
	with torch.no_grad():
		#vals = [true_pos, total_pos, true_neg, total_neg, correct, total] 
		vals = [0] * 6
		testing_size = len(testing_data)
		for i in tqdm(range(testing_size), disable=not debug):
			instance, label = testing_data[i]
			
			gene_vec = make_gene_vector(instance)
			expected = make_expected(label, label_to_ix)
			outputs = model(gene_vec)
			_, predicted = torch.max(outputs.data, 1)
			vals[5] += expected.size(0)
			vals[4] += (predicted == expected).item()
			
			if expected.item() == 1:  # getting normal is considered True, tumor is False
				vals[1] += 1
				if torch.equal(predicted, expected):
					vals[0] += 1  # true positive if predicted == actual
			elif expected.item() == 0:
				vals[3] += 1
				if torch.equal(predicted, expected):
					vals[2] += 1  # true negative if predicted == actual
			
	# account for specificity and sensitivity here
	stats = [vals[0] / vals[1], "", vals[2] / vals[3], "", vals[4] / vals[5]]
	vals = ["{:04}".format(i) for i in vals]
	names = ["Sensitivity:   ", "", "Specificity:   ", "", "Total Correct: "]
	if debug:
		for i in range(0, 6, 2):
			print(names[i], vals[i], "/", vals[i+1], "=", stats[i])
	if record_data: 
		if debug: 
			print("Recording data")
		f = open(percent_save_loc, "a")
		f.write(model.__str__() + "\t" + "\t".join(map(str, stats)) + "\n")
		f.close()

def test_models():
	start_time = time.monotonic()

	# Setting text files and other stuff to use
	key = {"full": (logged, 35728), "subset": (sub_logged, 4579)}
	text_file_normal = test_dir + data + "_test_normal_samples.txt"
	text_file_tumor = test_dir + data+ "_test_tumor_samples.txt"
	text_data, input_size = key[data]

	tumor_data_size, normal_data_size = 1994, 146
	testing_data = []
	add_to_data(testing_data, tumor_data_size, normal_data_size)
	random.shuffle(testing_data)

	models = [NN_zerow(), NN_dense(), NN_split()]
	
	for i, model in enumerate(models):
		model.load_state_dict(torch.load(stored_dict_locs[str(model)]))
		model.eval()
		if debug:
			print("\n", str(model) + " model")
		test_model(model, testing_data)
	
	if debug:
		print("\nRuntime:", timedelta(seconds=time.monotonic() - start_time))

if __name__ == "__main__":
	test_models()	
