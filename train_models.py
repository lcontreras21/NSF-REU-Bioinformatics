# Train the various models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
from settings import *
from datetime import timedelta
import random
from tqdm import tqdm
from models.process_data import *

def train_model(model, training_data):
	start_time = time.monotonic()
	# terminal message to track work
	if debug:
		print("Training", str(model), "model on the", data, "dataset with",
		tumor_data_size, "Tumor samples and", normal_data_size, "Normal samples.")
		print("Hyperparameters:",
			num_epochs, "epochs,",
			hidden_size, "neurons in the hidden layer,",
			learning_rate, "learning rate.")

	model = model.train()
	loss_function = nn.CrossEntropyLoss()
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

			gene_vec = make_gene_vector(instance)
			expected = make_expected(label, label_to_ix)

			# get probabilities from instance
			output = model(gene_vec)

			# apply learning to the model based on the instance
			loss = loss_function(output, expected)
			loss.backward()
			optimizer.step()

	if debug:
		print("\nSaving the model to file")
	torch.save(model.state_dict(), stored_dict_locs[str(model)])
	end_time = time.monotonic()
	if debug:
		print("Runtime:", timedelta(seconds=end_time - start_time), "\n")
