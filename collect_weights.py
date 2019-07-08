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
# no idea what I'm doing here

f = torch.load(dense_dict)
g = torch.load(split_dict)
h = torch.load(partial_dict)

# they are ordered dicts with weights and biases
# only want weights

dense = f["fc1.weight"]
partial = h["fc1.weight"]
split = []

for i in g:
	if "weight" in i:
		split.append(g[i])

# dense and partial are both tensors of size 50x4579
# split is a list of 50 tensors with various size 1xN

# now time to get weight sums for each node

# tensors of size 1x50
dense_ws = dense.sum(1)
partial_ws = partial.sum(1)

split_ws = torch.zeros(50)
for i, layer in enumerate(split[:-1]):
	total = layer.sum().item()
	split_ws[i] = total

def five_lines(data, names, function):
	special_nums = []
	for i in range(len(data)):
		ws = data[i]
		max_indices = []
		list_ws = ws.tolist()
		list_ws_dummy = deepcopy(list_ws)
		for ii in range(5):
			max_val = function(list_ws_dummy)
			list_ws_dummy.remove(max_val)
			max_indices.append(list_ws.index(max_val))
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
	for name, numbers in data:
		f.write(name +"\t" +"\t".join(list(map(str, numbers))) + "\n")
	f.close()

if debug:
	print("\nWeights")
data_to_write, special_nums = five_lines([dense_ws, partial_ws, split_ws], ["dense", "partial", "split"], max)

write_data(ws_save_loc, data_to_write)
write_data(w_save_loc, special_nums)


split_b = []
for i in g:
	if "bias" in i:
		split_b.append(g[i])
		split_bias = torch.zeros(50)
for i, layer in enumerate(split_b[:-1]):
	split_bias[i] = layer.item()
dense_bias = f["fc1.bias"]
partial_bias = h["fc1.bias"]
if debug:
	print("\nBiases")
data_to_write, special_nums = five_lines([dense_bias, partial_bias, split_bias], ["dense", "partial", "split"], max)

write_data(bs_save_loc, data_to_write)
write_data(b_save_loc, special_nums)

