'''
Train and test the models several times.
After each test, record the max weight indices
and save the results to a file.
'''

import torch
from settings import *
from itertools import combinations
from copy import deepcopy
import pickle
from operator import itemgetter

def pos_neg_sums(weight_data):
		# weight data is a tensor of size 50 x 4579
		sums = []
		for row in weight_data:
			pos_sum = row[row > 0].sum().item() 
			neg_sum = row[row < 0].sum().item()
			sums.append((pos_sum, neg_sum))
		return sums

### looking at weights and importance.
def collect_weights():
	names = ["Dense", "Zero-weights", "Split"]
	dicts = [torch.load(stored_dict_locs[model_name]) for model_name in names]
	### Input layer to Hidden Layer
	# weight_data = [dense, zerow, split]
	weight_data = [dicts[0]["fc1.weight"], dicts[1]["fc1.weight"], 
			[dicts[2][layer] for layer in dicts[2] if "weight" in layer]]

	with open(fc1_gene_weights_loc, "ab") as f:
		for i in zip(names, weight_data):
			pickle.dump(i, f)

	weights = [pos_neg_sums(weight_data[0]), pos_neg_sums(weight_data[1]), 
						[pos_neg_sums(layer)[0] for layer in weight_data[2][:-1]]]

	# Bias Values
	biases = [dicts[0]["fc1.bias"].tolist(), dicts[1]["fc1.bias"].tolist(), 
		[dicts[2][layer].item() for layer in list(dicts[2].keys())[:-2] if "bias" in layer]]
	
	all_fc1_biases = list(zip(names, biases))
	all_fc1_weights = list(zip(names, weights))
	write_data(fc1_weight_data_loc, all_fc1_weights)
	write_data(fc1_bias_vals_loc, all_fc1_biases)

	### Hidden Layer to Output layer
	weight_data = [i["fc2.weight"].tolist() for i in dicts]
	bias_data = [i["fc2.bias"].tolist() for i in dicts]
	write_data(fc2_weight_data_loc, list(zip(names, weight_data)))
	write_data(fc2_bias_data_loc, list(zip(names, bias_data)))

def process_gene_weights(nodes):
	model_names = ["Dense", "Split", "Zero-weights"]
	model_weight_data = {model_name:[] for model_name in model_names}
	with open(fc1_gene_weights_loc, "rb") as f:
		try:
			while True:
				line = pickle.load(f) #tuple(name, tensor(50x4579))
				name, gene_weights = line
				if name == "Split":
					desired_groups = list(itemgetter(*nodes)(gene_weights)) # keeps in order of nodes
				else:
					desired_groups = gene_weights[nodes] # keeps in order of nodes
				model_weight_data[name].append(desired_groups)
		except EOFError:
			pass
	model_node_data = {model_name:{node: [] for node in nodes} for model_name in model_names}
	for model in model_names:
		weight_data = model_weight_data[model]
		for node_datasets in weight_data:
			for i, dataset in enumerate(node_datasets):
				index = i % len(nodes)
				model_node_data[model][nodes[index]].append(node_datasets[index])
	return model_node_data

def process_n_weights(n=5):
	max_min_data = []
	duplicates = []
	with open(fc1_weight_data_loc, "r") as f:
		for i, line in enumerate(f):
			line = line.split("\t-\t")
			name, pdata = line[0], line[1].split("\t")

			pdata = [eval(i.replace("\n", "")) for i in pdata]
			unzipped_pdata = list(zip(*pdata)) # [pos values, neg values]

			pol_vals = [list(enumerate(i)) for i in unzipped_pdata]
			pol_vals.append(list(enumerate(pol_vals[0][i][1] + pol_vals[1][i][1] for i in range(len(pol_vals[0])))))
			#pol_vals contains [pos, neg, summed] weights

			top_weights = [list(list(zip(*sorted(i, key=itemgetter(1))[-n:]))[0]) for i in pol_vals]
			max_min_data.append([name, *top_weights])
			if i % 3 == 2:
				duplicates.append(capture_duplicates(max_min_data[-3:]))

	return max_min_data, duplicates

def process_n_biases(n=5):
	max_min_data = []
	duplicates = []
	with open(fc1_bias_vals_loc, "r") as f:
		for i, line in enumerate(f):
			line = line.split("\t-\t")
			name, bdata = line[0], line[1].replace("\n", "").split("\t")

			bdata = list(enumerate(map(float, bdata)))
			sorted_bdata = sorted(bdata, key=itemgetter(1))

			high_biases = list(list(zip(*sorted_bdata[-n:]))[0]) 
			low_biases = list(list(zip(*sorted_bdata[:n]))[0]) 
			
			max_min_data.append((name, high_biases, low_biases, []))
			if i % 3 == 2:
				duplicates += capture_duplicates(max_min_data[-3:])

	return max_min_data, duplicates
	
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
		duplicates += [name, pos_intersect, neg_intersect, summed_intersect]
	return duplicates

def write_data(file_loc, datasets):
	with open(file_loc, "a") as f:
		for dataset in datasets:
			name = dataset[0]
			values = [list(map(str, i)) for i in dataset[1:]]
			string = ""
			for i in values:
				string += "\t-\t" + "\t".join(i)
			f.write(name + string + "\n") 

if __name__ == "__main__":
	collect_weights()
