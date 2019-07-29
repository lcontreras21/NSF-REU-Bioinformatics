# Test the dense model with different number of hidden nodes
# Going from 1, to several magnitudes higher

from models.nn_dense import NN_dense
from models.process_data import load_training_data
from analyze import *
from test_models import *
from collect_weights import *
import time
from datetime import timedelta
import sys

def print_dense_percents(size, time_to_run):
	percents = [0] * 3 
	total = 0
	with open(percent_save_loc, "r") as f:
		for line in f:
			total += 1
			line = line.split()
			run_data = list(map(float, line[1:]))
			percents = [percents[i] + run_data[i] for i in range(3)]
	percents = ["{0:.9f}".format(percents[i] / total) for i in range(3)]
	print(size, *percents, time_to_run, sep="\t")


def test_dense():
	hidden_sizes = [1, 10, 25, 50, 75, 100, 150, 300, 500]
	training_data = load_data("train")
	testing_data = load_data("test")
	for i, size in enumerate(hidden_sizes):
		start_size_time = time.monotonic()
		set_starting_seed(size)
		nn_dense = NN_dense(size)
		for test in range(3):
			train_model(nn_dense, training_data)
			test_models(testing_data, models=[NN_dense(size)])
		end_size_time = time.monotonic()
		print_dense_percents(size, timedelta(seconds=end_size_time - start_size_time))
		reset_files()

if __name__ == "__main__":
	test_dense()
