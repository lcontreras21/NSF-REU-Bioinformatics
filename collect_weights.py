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
	# they are ordered dicts with weights and biases
	dicts = [torch.load(dense_dict), torch.load(partial_dict), torch.load(split_dict)]

	# weight_data = [dense, partial, split]
	weight_data = [dicts[0]["fc1.weight"], dicts[1]["fc1.weight"], []]

	for layer in dicts[2]:
		if "weight" in layer: 
			weight_data[2].append(dicts[2][layer])

	# now time to get weight sums for each node
	weights = [weight_data[0].sum(1), weight_data[1].sum(1), torch.zeros(50)]
	for i, layer in enumerate(weight_data[2][:-1]):
		total = layer.sum().item()
		weights[2][i] = total

	# bias = [dense, partial, split"]
	biases = [dicts[0]["fc1.bias"], dicts[1]["fc1.bias"], torch.zeros(50)]

	split_b = []
	for i in dicts[2]:
		count = 0
		if "bias" in i:
			split_b.append(dicts[2][i])	
	for i, layer in enumerate(split_b[:-1]):
		biases[2][i] = layer.item()
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
