# Test the model or models 
# Import model from saved file

import torch
import torch.nn as nn
import time
from datetime import timedelta
from tqdm import tqdm
from fractions import Fraction

from models.process_data import * 
from models.nn_zerow import NN_zerow
from models.nn_split import NN_split
from models.nn_dense import NN_dense
from settings import *
import sys

def test_models(testing_data, models):
	start_time = time.monotonic()
	for i, model in enumerate(models):
		model.load_state_dict(torch.load(stored_dict_locs[str(model)]))
		if debug: 
			print("Testing accuracy of", model, "model using the", data, "dataset")
		
		model.eval()
		with torch.no_grad():
			keys = [1, 0, "total"]
			vals = {1: [0, 0], 0: [0, 0], "total": [0, 0]}
			testing_size = len(testing_data)
			for i in tqdm(range(testing_size), disable=not debug):
				instance, label = testing_data[i]
				
				gene_vec = make_gene_vector(instance)
				expected = make_expected(label, label_to_ix)
				outputs = model(gene_vec)
				#_, predicted = torch.max(outputs.data, 1)
				predicted = outputs.data.round()
				vals["total"][1] += 1
				
				vals[expected.item()][1] += 1 # Account for total pos or neg
				if torch.equal(predicted, expected):
					vals[expected.item()][0] += 1 # Account for true pos or neg
					vals["total"][0] += 1
				
		# account for specificity and sensitivity here
		stats = [float(Fraction(*vals[i])) for i in keys]
		formatted = [["{:04}".format(i) for i in vals[key]] for key in keys]
		name = ["Sensitivity", "Specificity", "Correctness"]
		if debug:
			for i in range(3):
				print(name[i], "/".join(formatted[i]), "=", "{:04}".format(stats[i]))

		if record_data: 
			if debug: 
				print("Recording data")
			with open(percent_save_loc, "a") as f:
				f.write(model.__str__() + "\t" + "\t".join(map(str, stats)) + "\n")
	
	if debug:
		print("\nRuntime:", timedelta(seconds=time.monotonic() - start_time))

def user_test_input():
	testing_data = load_data("test")
	keys = {"split": NN_split(), "dense": NN_dense(), "zerow": NN_zerow()}
	if len(sys.argv[1:]) == 0:
		test_models(testing_data, [NN_dense(), NN_split(), NN_zerow()])	
	elif sys.argv[1].lower() == "all":
		test_models(testing_data, [NN_dense(), NN_split(), NN_zerow()])	
	elif len(sys.argv[1:]) >= 1:
		for i in sys.argv[1:]:
			if i.lower() not in keys:
				print("Wrong model name:", i)
				return
		test_models(testing_data, [keys[model] for model in sys.argv[1:]]) 


if __name__ == "__main__":
	debug = True
	user_test_input()
