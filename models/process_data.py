### Place to store helper functions that load and process the text data
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from settings import *
import pickle
from collections import defaultdict

# adds data to a list for training or testing. Could add an
# even amount or uneven amount based on inputs.
def add_to_data(data_set, text_files):
	for text_file in text_files:
		with open(text_file, "r") as f:
			for line in f:
				line = line.split()
				data_set += [(line[1:-1], line[-1])]

def load_data(mode):
	dirs = {"train": train_dir, "test": test_dir}
	if debug: 
		print("Loading", mode, "data")
	data_set = []
	text_files = [dirs[mode] + data + "_" + mode + "_" + i + "_samples.txt" for i in ["normal", "tumor"]]
	add_to_data(data_set, text_files)
	return data_set

# to be used for determining output of NN,
# essentially our sigmoid function but other option is possible
label_to_ix = {"Tumor": [1], "Normal": [0]}
def make_expected(label, label_to_ix):
	return torch.FloatTensor([label_to_ix[label]])

# transforms a the gene list into a tensor for our NN
def make_gene_vector(input_sample):
	processed = list(map(float, input_sample))
	vec = torch.FloatTensor(processed)
	return vec.view(1, -1)

# makes a dictionary for gene index to be able to connect nodes
def gene_dict(data=data):
	text_data = "text_files/" + data + "_gene_names.txt"
	with open(text_data, "r") as f:
		gene_names = f.readline().split()

	gene_to_index = defaultdict(list)
	for index, key in enumerate(gene_names[1:-1]):
		gene_to_index[key].append(index)
	return dict(gene_to_index)

# import hallmark gene data
def import_gene_groups():
	with open(text_gene_groups, "r") as f:
		gene_groups = [line.split()[2:] for line in f]
	return gene_groups

def get_gene_indicies(gene_group, gene_indexes):
	with open(gene_pairs_loc, "rb") as f:
		alternate_names = pickle.load(f)
	indices = []
	for gene in gene_group:
		try:
			if gene in alternate_names.keys():
				gene = alternate_names[gene]
			indices += gene_indexes[gene]
		except:
			pass
	indices.sort()
	return indices

def save_indicies():
	gene_groups = import_gene_groups()
	gene_indexer = gene_dict()
	with open("text_files/gene_indicies.txt", "w") as f:
		for gene_group in gene_groups:
			x = get_gene_indicies(gene_group, gene_indexer)
			f.write("\t".join(list(map(str, x))) + "\n")
def read_indicies():
	gene_indicies = []
	with open("text_files/gene_indicies.txt", "r") as f:
		for line in f:
			line = list(map(int, line.split("\t")))
			gene_indicies.append(line)
	return gene_indicies

def set_starting_seed(hidden_size=hidden_size):
	input_layer = nn.Linear(input_size, hidden_size)
	input_layer.weight.data.normal_(0.0, 1/(input_size**(0.5)))

	output_layer = nn.Linear(hidden_size, output_size)
	output_layer.weight.data.normal_(0.0, 1/(hidden_size**(0.5)))
	with open(starting_seed_loc, "wb") as f:
		pickle.dump(input_layer.weight.data, f) 
		pickle.dump(output_layer.weight.data, f)

def get_starting_seed():
	with open(starting_seed_loc, "rb") as f:
		input_seed = pickle.load(f)
		output_seed = pickle.load(f)
	return input_seed, output_seed


# collect information on the heaviest weights that were removed
def weight_info(interesting_weights=weights_to_test):
	with open(text_gene_groups, "r") as f:
		info = [] # tuple (index, gene group name, number of genes in group)
		for index, line in enumerate(f):
			if index in interesting_weights:
				line = line.split()
				info.append((index, line[0], len(line[2:])))
		for i in info:
			print(i)

# finds the number of unique genes in each gene group
def gene_groups_info():
	f = open(text_gene_groups, "r")
	info = []
	for line in f:
		info.append(len(line.split()[2:]))
	counts = {i:info.count(i) for i in set(info)}
	x = list(counts.keys())
	x.sort()
	for i in list(x):
		print(i, counts[i])

def reset_files():
	files = [fc1_weight_data_loc, fc1_bias_vals_loc, fc1_gene_weights_loc, fc2_weight_data_loc, fc2_bias_data_loc, percent_save_loc]
	for f in files:
		open(f, "w").close()
	open(fc1_gene_weights_loc, "wb").close()

if __name__ == "__main__":
	reset_files()

