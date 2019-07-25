'''
Open the saved data from the files and analyze them
Check if numbers are repeated and how many times there are overlap
'''
from settings import *
from copy import deepcopy
from collections import Counter
from operator import itemgetter
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
from collect_weights import *

def process_data():
	wd, weights = process_n_weights(5)
	bd, biases = process_n_biases(5)
	return [wd, weights, bd, biases]

def normalize(d):
	# d is numpy array
	total = d.sum()
	if total == 0: 
		total = 1
	d = d / total
	return d

def make_gene_distributions(datasets, data_type, normalized=False):
	'''
	Parameter information:
		normalized can be True, False, or "both"
		datasets = [weight_overlap, weights, bias_overlap, bias]
		each dataset in datasets is list(name, max, min, summed)
	'''
	name_key = {"Split": 0, "Dense": 1, "Zero-weights": 2,
			"Dense_Zero-weights": 3, "Dense_Split": 4, "Zero-weights_Split": 5}
	#dists [contains three lists for max, min, summed with
	#   [each of the following being a dict: {split}, {dense}, {zerow}, {d-p}, {d-s}, {p-s}
	#   recording the statistics for each weight]
	dists = [[np.zeros(50, dtype=int) for j in range(6)] for i in range(3)]
	normalized_dists = deepcopy(dists)
	if data_type == "weights":
		x = datasets[0] + datasets[1]
	elif data_type == "biases":
		x = datasets[2] + datasets[3]
		for dataset in x:
			print(dataset)
			name_index = name_key[dataset[0]]
			dataset = dataset[1:]
			for polarity in range(3):
				for value in dataset[polarity]:
					# access the max, min, or summed data
					# then access which model the data belongs to
					# then keep track of the weight value in the dictionary
					dists[polarity][name_index][value] += 1
					normalized_dists[polarity][name_index] = normalize(dists[polarity][name_index])

					if normalized == "both":
						return dists, normalized_dists
					if normalized:
						return normalized_dists
					return dists

# Three types of data: dense, split, zero-weights
def draw_gene_graph(dists, title, save_location="diagrams/distribution.pdf"):
	# dists is list of numpy arrays with info [split, dense, zerow, d-p, d-s, p-s]i
	names = ["Split Model", "Dense Model", "Zero-weights Model", 
			"Dense-Zero Overlap", "Dense-Split Overlap", "Zero-Split Overlap"]
	colors = ["r", "g", "b", "tab:grey", "tab:brown", "tab:purple"]
	
	fig, axs = plt.subplots(3, 2, sharey=True)
	axs = axs.tolist()
	axs_list = [inner for outer in axs for inner in outer]
	fig.suptitle("Distribution of " + title)
	fig.subplots_adjust(hspace=0.5)
	if test_behavior:
		weights_to_test.sort()
		text =  title.split()[-1] + " removed are " + ", ".join(map(str, weights_to_test))
	else:
		text = "No weights removed"
	plt.figtext(0.5, 0.02, text, ha="center", va="bottom")
	plt.rcParams['xtick.labelsize'] = 4
	for index, ax in enumerate(axs_list):
		current_dist = dict(enumerate(dists[index]))
		ax.bar(current_dist.keys(), current_dist.values(), color=colors[index], align='center')
		ax.set_title(names[index])
		ax.set_xticklabels(labels=current_dist.keys(), minor=True, rotation='vertical')
	
	plt.savefig(save_location)
def draw_gene_graphs(which="both"):
	# which can be one of {'both', 'weights', biases'}
	processed_data = process_data()
	key = {"weights":["weights"], "biases":["biases"], "both": ["weights", "biases"]}

	for data_type in key[which]:
		unnorm, norm = make_distributions(processed_data, data_type, normalized="both")
		data_pair = [(unnorm, "_unnormalized"), (norm, "_normalized")]
		polarity_names = ["Positive", "Negative", "Summed"]
		for data_style, name in data_pair:
			if data_type == "biases":
				polarity_names = ["Positive", "Negative"]
				data_style = data_style[:2]
				for i, polarity_data in enumerate(data_style):
					graph_name = "Top 5 " + polarity_names[i] + " " + data_type.capitalize()
					save_name = image_path + data_type + "_" + polarity_names[i].lower() + name + modded + ".pdf"
					draw_graph(polarity_data, graph_name, save_location=save_name)

def gene_statistics(n):
	#weights_to_test = [26, 27, 19, 34, 25, 3, 8, 13, 41]
	weights_to_test = [2]
	model_node_data = process_gene_weights(weights_to_test)
	model_names = ["Split", "Dense", "Zero-weights"]
	top_gene_dists = {model_name:{node:[] for node in weights_to_test} for model_name in model_names}

	for model_name in model_names:
		print(model_name)
		for node in model_node_data[model_name]:
			print(node)
			for x in model_node_data[model_name][node]:
				x = list(enumerate(x.tolist()))
				sorted_x = sorted(x, key=itemgetter(1))
				top_x = list(list(zip(*sorted_x[-n:]))[0])
				min_x = list(list(zip(*sorted_x[:n]))[0])
				top_gene_dists[model_name][node].append()

				def order_dists(dists, n):
					# Order the distributions to see the heaviest weights.
					polarity_names = ["Positive", "Negative", "Summed"]
					datatype_names = ["Split", "Dense", "Zero-weights", "Dense_Zero-weights", "Dense_Split", "Zero-weights_Split"]
					#datatype_names = ["Split", "Zero-weights"]
					for i, polarity in enumerate(dists):
						print(polarity_names[i])
						#polarity = [polarity[0], polarity[2]]
						for j, datatype in enumerate(polarity[:3]):
							datatype = dict(enumerate(datatype))
							sorted_dt = list(zip(*sorted(datatype.items(), key=itemgetter(1), reverse=True)[:n]))[0]
							sorted_dt = ["{:02}".format(i) for i in sorted_dt]
							print("{:12}".format(datatype_names[j]), sorted_dt)
							print()

if __name__ == "__main__":
	pass
