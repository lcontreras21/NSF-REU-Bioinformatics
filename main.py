### instead of bash script, run files here

from nn_partial_links import *
from nn_dense import *
from nn_split import *
from test_models import *
from collect_weights import *
from analyze import *
import sys
import time
from datetime import timedelta

def analyze(n):
	status = {True: "on", False: "off"}
	print("Before starting data collection, are these settings correct?")
	print("Debug is", status[debug], 
		"\b, Data recording is", status[record_data], 
		"\b, Testing weights is", status[test_behavior],
		"\b, And using the " + data + " dataset")
	correct = input("[Y, n] ")
	if correct.lower() in ["yes", "y", "ye"]:
		for i in range(3):
			print("\033[F", end= "\r")
			print(" " * 100, end="\r")
		pass	
	else:
		print("Aborting...")
		return
	
	start_time = time.monotonic()
	# Testing and training models, Data Collection loop
	print("|Settings| Debugging is", status[debug],
			"\b, Data recording is", status[record_data],
			"\b, Testing weight environment is", status[test_behavior])
	print("\nStarting Data Collection")
	for i in range(int(n)):
		print("Currently on iteration", "{:03}".format(i+1), end="\r")
		# Trains the models
		train_partial_model()
		train_dense_model()
		train_split_model()

		# Records the weights and saves to file
		collect_weights()

		# Tests the models and saves percentages to file
		test_models()
	loop_time = time.monotonic()
	print(" " * 50, end="\r")
	print("\033[F-----Finished", n, "tests in", timedelta(seconds=loop_time - start_time))

	# Analysis Portion
	if test_behavior:
		print("\nChecking if weights were properly removed")
		verify_removed_weights()
	
	print("\nPercentages from session")
	print_percentages()

	print("\nDrawing distributions and saving to files under /diagrams/")
	draw_graphs(which="both")

	end_time = time.monotonic()
	print("\n-----Finished experiment in", timedelta(seconds=end_time - start_time))
	

if __name__ == "__main__":
	analyze(sys.argv[1])

