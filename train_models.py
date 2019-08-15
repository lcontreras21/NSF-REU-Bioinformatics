# Train the various models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.process_data import *
from models.nn_split import *
from models.nn_dense import *
from models.nn_zerow import *

from settings import *

import time
from datetime import timedelta
import random
from tqdm import tqdm
import sys


def train_model(model, training_data):
	start_time = time.monotonic()
	# terminal message to track work
	if debug:
		print("Training", str(model), "model on the", data, "dataset with",
		7975, "Tumor samples and", 584, "Normal samples.")
		print("Hyperparameters:",
			num_epochs, "epochs,",
			hidden_size, "neurons in the hidden layer,",
			learning_rate, "learning rate.")

	model = model.train()
	loss_function = nn.BCELoss()
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

	# train the model
	if debug:
		print("Training the model")
	for epoch in tqdm(range(num_epochs), disable=not debug):
		random.shuffle(training_data)
		for i in tqdm(range(len(training_data)), disable=not debug):
			# erase gradients from previous run
			instance, label = training_data[i]
			model.zero_grad()

			gene_vec = make_gene_vector(instance) # convert input sample to tensor
			expected = make_expected(label, label_to_ix) # converts normal/tumor to tensor

			# get probabilities from instance
			output = model(gene_vec)

			# apply learning to the model based on the instance
			loss = loss_function(output, expected)
			loss.backward()
			optimizer.step()

			model.mask() # make sure weights are still zeroed for certain models

	if debug:
		print("\nSaving the model to file")
	torch.save(model.state_dict(), stored_dict_locs[str(model)])
	end_time = time.monotonic()
	if debug:
		print("Runtime:", timedelta(seconds=end_time - start_time), "\n")

def user_train_input():
	training_data = load_data("train")
	keys = {"split": NN_split(), "dense": NN_dense(), "zerow": NN_zerow()}
	
	if len(sys.argv[1:]) >= 1:
		if sys.argv[2] == "all":
			for model in keys:
				train_model(keys[model], training_data)
				return
		else:
			for i in sys.argv[1:]:
				if i.lower() not in keys:
					raise ValueError("Wrong model name:", i)
					return
			for i in sys.argv[1:]:
				train_model(keys[i.lower()], training_data)
			return
	else:
		for model in keys:
			train_model(keys[model], training_data)
			return

if __name__ == "__main__":
	debug = True
	user_train_input()
