'''
Train and test the models several times.
After each test, record the max weight indices
and save the results to a file.
'''

import torch
from settings import *
from itertools import combinations
from copy import deepcopy
import operator
import pickle

def pos_neg_sums(weight_data):
	# weight data is a tensor of size 50 x 4579
	sums = []
	for row in weight_data:
		pos_sum = row[row > 0].sum().item() 
		neg_sum = row[row < 0].sum().item()
		sums.append((pos_sum, neg_sum))
	return sums

### looking at weights and importance.
def process():
	names = ["Dense", "Zero-weights", "Split"]
	dicts = [torch.load(stored_dict_locs[model_name]) for model_name in names]

	# weight_data = [dense, zerow, split]
	weight_data = [dicts[0]["fc1.weight"], dicts[1]["fc1.weight"], 
			[dicts[2][layer] for layer in dicts[2] if "weight" in layer]]

	with open(all_gene_weights_loc, "ab") as f:
		for i, weights in enumerate(weight_data):
			pickle.dump((names[i], weights), f)

	weights = [pos_neg_sums(weight_data[0]), pos_neg_sums(weight_data[1]), 
			[pos_neg_sums(layer)[0] for layer in weight_data[2][:-1]]]

	# Bias Values
	biases = [dicts[0]["fc1.bias"], dicts[1]["fc1.bias"], 
		torch.FloatTensor([dicts[2][layer].item() for layer in list(dicts[2].keys())[:-2] if "bias" in layer])]
	biases = [pos_neg_sums(i) for i in biases]	
	return weights, biases #returns a list of list of tuples of pos and negative sums
	

def capture_duplicates(special_numbers):
	combos = list(combinations(special_numbers, 2))	
	duplicates = []

	# special_numbers is a list of tuples (name, max/min numbers)
	for pair in combos:
		first, second = pair
		pos_intersect = list(set(first[1]).intersection(second[1]))
		neg_intersect = list(set(first[2]).intersection(second[2]))
		summed_intersect = list(set(first[3]).intersection(second[3]))
		name = "{}".format(first[0] + "_" + second[0])
		if debug:
			pname = "{:<18}".format(first[0] + "_" + second[0])
			print(pname, "Pos Intersect:", pos_intersect)
			print(pname, "Neg Intersect:", neg_intersect)
			print(pname, "Combined Intersect", summed_intersect)
		duplicates.append((name, pos_intersect, neg_intersect, summed_intersect))
	return duplicates

def max_min_data(datasets, n):
	names = ["Dense", "Zero-weights", "Split"]
	max_mins_summed = []
	index = 0
	for dataset in datasets:
		top_summed_n = []
		summed_values = list(enumerate(i + j for i, j in dataset))
		top_summed_n = list(zip(*sorted(summed_values, key=operator.itemgetter(1))[-n:]))[0]
		separate_values = list(zip(*dataset))
		pos_values = list(enumerate(separate_values[0]))
		neg_values = list(enumerate(separate_values[1]))
		top_pos_n = list(zip(*sorted(pos_values, key=operator.itemgetter(1))[-n:]))[0]
		top_neg_n = list(zip(*sorted(neg_values, key=operator.itemgetter(1))[-n:]))[0]
		max_mins_summed.append((names[index], top_pos_n, top_neg_n, top_summed_n))
		index += 1
	return max_mins_summed

def specialized_data(datasets, n):
	special_nums = max_min_data(datasets, n)
	duplicates = capture_duplicates(special_nums)	
	return duplicates, special_nums

def write_data(file_loc, datasets):
	with open(file_loc, "a") as f:
		for dataset in datasets:
			name = dataset[0]
			values = [list(map(str, i)) for i in dataset[1:]]
			string = ""
			for i in values:
				string += "\t-\t" + "\t".join(i)
			f.write(name + string + "\n") 

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

	names = ["Dense", "Zero-weights", "Split"]
	all_weights = list(zip(names, weights))
	write_data(all_weight_data_loc, all_weights)

if __name__ == "__main__":
	collect_weights()
