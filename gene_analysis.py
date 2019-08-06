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
from models.process_data import *

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
def draw_gene_graph(dists, title, save_location):
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
		unnorm, norm = make_gene_distributions(processed_data, data_type, normalized="both")
		data_pair = [(unnorm, "_unnormalized"), (norm, "_normalized")]
		polarity_names = ["Positive", "Negative", "Summed"]
		for data_style, name in data_pair:
			if data_type == "biases":
				polarity_names = ["Positive", "Negative"]
				data_style = data_style[:2]
				for i, polarity_data in enumerate(data_style):
					graph_name = "Top 5 " + polarity_names[i] + " " + data_type.capitalize()
					save_name = image_path + data_type + "_" + polarity_names[i].lower() + name + ".pdf"
					draw_gene_graph(polarity_data, graph_name, save_location=save_name)

# Given the desired hallmark sets from analyze.py, go in depth and find details about individual genes.

def input_analysis(nodes, cutoff=0.06):
	model_node_data = process_gene_weights(nodes)
	gene_indicies = read_indicies()
	# model_node_data is a dict of each model to a dict of each node and its line of weights
	# for each model
	#	for each node (nodes will have different data lengths based on gene groups)
	#		for each gene weight in the gene group, make sure to label with name too to keep track
	#			keep track of neg, pos, avg neg, avg pos, <neg cutoff, > neg cutoff, neg avg cutoff, pos avg cutoff
	gene_data = {model_name:
					{hidden_node:
						{gene_node:[0]*8 
						for gene_node in range(len(gene_indicies[hidden_node]))} 
						for hidden_node in nodes} 
					for model_name in model_node_data}
	
	# avg pos and negative weight are around 0.063
	for model_name in model_node_data:
		for hidden_node in model_node_data[model_name]:
			for input_weights in model_node_data[model_name][hidden_node]:
				if model_name != "Split":
					input_weights = input_weights[gene_indicies[hidden_node]]
				assert(len(input_weights) == len(gene_indicies[hidden_node]))
				for index, gene_weight in enumerate(input_weights):
					gene_weight = gene_weight.item()
					node_data = gene_data[model_name][hidden_node][index]
					if gene_weight > 0:
						node_data[1] += 1
						node_data[5] += gene_weight
						if gene_weight > cutoff:
							node_data[3] += 1
							node_data[7] += gene_weight
					elif gene_weight < 0:
						node_data[0] += 1
						node_data[4] += gene_weight
						if gene_weight < -cutoff:
							node_data[2] += 1
							node_data[6] += gene_weight
			node_data = gene_data[model_name][hidden_node]
			for gene_node in node_data:
				for i in range(3):
					try:
						node_data[gene_node][i+4] /= node_data[gene_node][i]
					except:
						pass
	return gene_data	

def crit_analysis(node, cutoff=0.06):
	gene_data = input_analysis(node, cutoff)
	gene_names = import_gene_groups()

	count = 0
	for model_name in ["Split"]:
		for hidden_node in gene_data[model_name]:
			count = 0
			for gene_node in gene_data[model_name][hidden_node]:
				node_data = gene_data[model_name][hidden_node][gene_node]
				node_names = gene_names[hidden_node]
				x = node_data[2] + node_data[3]
				if x > 80:
					count += 1
					print(node_names[gene_node], *node_data[:4], sep="\t")
	print(count)
if __name__ == "__main__":
	crit_analysis([9], cutoff=0.05)
	#input_analysis([9], cutoff=0.01)
	pass
