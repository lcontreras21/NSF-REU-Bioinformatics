'''
Train and test the models several times.
After each test, record the max weight indices
and save the results to a file.
'''

import torch
from settings import *
from itertools import combinations
from copy import deepcopy

### looking at weights and importance.
def process():
	name_list = ["Dense", "Zero-weights", "Split"]
	dicts = [torch.load(stored_dict_locs[model_name]) for model_name in name_list]

	# weight_data = [dense, partial, split]
	weight_data = [dicts[0]["fc1.weight"], dicts[1]["fc1.weight"], []]
	# adding data from split model
	weight_data[2] = [dicts[2][layer] for layer in dicts[2] if "weight" in layer]

	### Weight Sums
	weights = [weight_data[0].sum(1), weight_data[1].sum(1), torch.zeros(50)]
	# summing all data from split model
	for i, layer in enumerate(weight_data[2][:-1]):
		weights[2][i] = layer.sum().item()

	# Bias Sums
	biases = [dicts[0]["fc1.bias"], dicts[1]["fc1.bias"], torch.zeros(50)]
	biases[2] = torch.FloatTensor([dicts[2][layer].item() for layer in list(dicts[2].keys())[:-2] if "bias" in layer])

	return weights, biases

def five_lines(data, names, function):
	special_nums = []
	for i in range(len(data)):
		ws = data[i].tolist()
		max_indices = []
		ws_copy = deepcopy(ws)
		for ii in range(5):
			special_val = function(ws_copy)
			ws_copy.remove(special_val)
			max_indices.append(ws.index(special_val))
		max_indices.sort()
		if debug:
			print("{:<8}".format(names[i]), max_indices)
		special_nums.append((names[i], max_indices))
		
	combos = list(combinations(special_nums, 2))	
	duplicates_to_write = []
	for pair in combos:
		intersect = list(set(pair[0][1]).intersection(pair[1][1]))
		name = "{:<15}".format(pair[0][0] + "-" + pair[1][0])
		if debug:
			print(name, "like numbers", intersect)
		duplicates_to_write.append((name, intersect))
	return duplicates_to_write, special_nums

def write_data(ffile, data):
	f = open(ffile, "a")
	for name, values in data:
		f.write(name +"\t" +"\t".join(list(map(str, values))) + "\n")
	f.close()

def collect_weights():
	name_list = ["dense", "partial", "split"]
	weights, biases = process()

	if debug:
		print("\nWeights")
	data_to_write, special_nums = five_lines(weights, name_list, max)

	write_data(ws_save_loc, data_to_write)
	write_data(w_save_loc, special_nums)

	if debug:
		print("\nBiases")
	data_to_write, special_nums = five_lines(biases, name_list, max)

	write_data(bs_save_loc, data_to_write)
	write_data(b_save_loc, special_nums)

if __name__ == "__main__":
	collect_weights()
