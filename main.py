'''
Main file for the project
If running from a python terminal, only need to import this file. 
Then, functions from other files can then be run. 
'''

from models.nn_zerow import *
from models.nn_dense import *
from models.nn_split import *
from test_models import *
from models.process_data import set_starting_seed
from other_py_files.make_subset_data import *
from train_models import *
from collect_weights import *
from analyze import *

import sys
import time
from datetime import timedelta

def create_files():
	print("Creating subset, train, test, and other files")
	print("Estimated time to finish: 00:01:30")
	start = time.monotonic()
	# first need to create the subset data set from logged_scaled_rnaseq.txt
	save_gene_names()
	make_subset_dataset()

	# then need to create gene_names files for subset and the full set
	save_gene_names("subset_")

	# create separate normal and tumor sets for subset and full set
	norm_tum_sets()

	# now we need to create the training and testing data for both sets
	create_train_test_data()

	# lastly create file for gene indicies and other files
	save_indicies()
	set_starting_seed()

	end = time.monotonic()
	print("Run time:", timedelta(seconds=end-start))

def main(n, models=["dense", "split", "zerow"]):
	presets = [debug, record_data, test_behavior, seed, data]
	message = ["|Settings| Debugging is ", "Data recording is ", "Testing weight environment is ", "Fixed starting seed is ", "Dataset is "]
	status = {True: "on", False: "off", "subset": "subset dataset", "full": "full dataset"}
	settings_message = ["{}{}".format(b, status[a]) for a, b in zip(presets, message)] 
	
	print("Before starting data collection, are these settings correct?")
	print(*settings_message, sep=", ")
	correct = input("[Y, N] ")
	if correct.lower() in ["yes", "y", "ye"]:
		pass	
	else:
		print("Aborting...")
		return
	
	start_time = time.monotonic()

	#print("Creating a new randomized subset of data for training and testing.")
	#create_train_test_data() # remove this function if randomized data isn't needed
	
	training_data = load_data("train")
	testing_data = load_data("test")
	
	print(" " * 50, end="\r")
	set_starting_seed()	
	
	print("\nStarting Data Collection")
	for i in range(int(n)):
		# Trains the models
		status = "Currently on iteration " + "{:03}".format(i+1) + ", working on:"
		model_key = {"dense": NN_dense(), "split": NN_split(), "zerow": NN_zerow()}
		for i in models:
			print(status, i, end="\r")
			train_model(model_key[i.lower()], training_data)

		# Records the weights and saves to file
		if record_data:
			collect_weights()

		# Tests the models and saves percentages to file
		test_models(testing_data, [NN_zerow(), NN_dense(), NN_split()])
	loop_time = time.monotonic()
	print(" " * 50, end="\r")
	print("\033[F-----Finished", n, "tests in", timedelta(seconds=loop_time - start_time))

	# Analysis Portion
	if test_behavior:
		print("\nChecking if weights were properly removed")
		verify_removed_weights()
	
	print("\nPercentages from session")
	print_percentages()

	if testing_parameters: # if testing parameters and don't want to save data
		reset_files()

	end_time = time.monotonic()
	print("\n-----Finished experiment in", timedelta(seconds=end_time - start_time))

def user_main_input():
	if sys.argv[1] == "create_files":
		create_files()
	else:
		if len(sys.argv[1:]) > 1:
			if sys.argv[2] == "all":
				main(sys.argv[1])
				return
			else:
				for i in sys.argv[2:]:
					if i.lower() not in ["split", "dense", "zerow"]:
						raise ValueError(i + " not a valid model name.")
						return
				main(sys.argv[1], models=sys.argv[2:])
				return
		else:
			main(sys.argv[1])
			return


if __name__ == "__main__":
	user_main_input()
