### instead of bash script, run files here

from nn_partial_links import *
from nn_dense import *
from nn_split import *
from test_models import *
from collect_weights import *
import sys

def analyze(n):
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

if __name__ == "__main__":
	analyze(sys.argv[1])

