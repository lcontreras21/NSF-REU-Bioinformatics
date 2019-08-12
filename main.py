### instead of bash script, run files here

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

def main(n):
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
	
	#create_train_test_data()
	save_indicies()
	training_data = load_data("train")
	testing_data = load_data("test")
	
	
	print(" " * 50, end="\r")
	set_starting_seed()	
	
	print("\nStarting Data Collection")
	for i in range(int(n)):
		# Trains the models
		status = "Currently on iteration " + "{:03}".format(i+1) + ", working on:"
		
		print(status, "Zerow", end="\r")
		train_model(NN_zerow(), training_data)

		print(status, "Dense", end="\r")
		train_model(NN_dense(), training_data)

		print(status, "Split", end="\r")
		train_model(NN_split(), training_data)

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

	#if record_data:
	#	print("\nDrawing distributions and saving to files under", image_path)
	#	draw_graphs(which="both")

	if testing_parameters:
		reset_files()

	end_time = time.monotonic()
	print("\n-----Finished experiment in", timedelta(seconds=end_time - start_time))
	

if __name__ == "__main__":
	if sys.argv[1] == "create_files":
		create_files()
	else:
		main(sys.argv[1])
