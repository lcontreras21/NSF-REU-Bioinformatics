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

	# weight_data = [dense, zerow, split]
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

def specialized_data(datasets, n):
	names = ["dense", "zerow", "split"]
	special_nums = [] # contains tuples of name and weights
	for i, dataset in enumerate(datasets): # data is three model weights
		dataset = dataset.tolist()
		max_indices = []
		dataset_copy = deepcopy(dataset)
		for ii in range(n): # find the top five weights
			special_val = max(dataset_copy)
			dataset_copy.remove(special_val)
			max_indices.append(dataset.index(special_val))
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

def write_data(file_loc, data):
	with open(file_loc, "a") as f:
		for name, values in data:
			f.write(name +"\t" +"\t".join(list(map(str, values))) + "\n")

def collect_weights():
	weights, biases = process()

	if debug:
		print("\nWeights")
	duplis, special_nums = specialized_data(weights, 5)

	write_data(ws_save_loc, duplis)
	write_data(w_save_loc, special_nums)

	if debug:
		print("\nBiases")
	duplis, special_nums = specialized_data(biases, 5)

	write_data(bs_save_loc, duplis)
	write_data(b_save_loc, special_nums)

	names = ["dense", "zerow", "split"]
	weight_nums = []
	for weight in weights:
		weight_nums.append([x.item() for x in weight])
	paired_data = [(names[i], weight_nums[i]) for i in range(3)]
	write_data(all_weight_data_loc, paired_data)

if __name__ == "__main__":
	collect_weights()
