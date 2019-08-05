'''
Analyze weights from the hidden layer to the output layer
'''
from settings import *
from collect_weights import * 
from operator import itemgetter
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from pprint import pprint

def load_output_data():
	file_dir = "text_files/analysis/swapped/"
	fc2_weight_data_loc = file_dir + "fc2_weights.txt"
	fc2_bias_data_loc = file_dir + "fc2_bias.txt"
	files = [fc2_weight_data_loc, fc2_bias_data_loc] 
	weight_data = []
	bias_data = []
	for index, file_i in enumerate(files):
		with open(file_i, "r") as f:
			for line in f:
				line = line.split("\t-\t")
				line[1] = line[1].replace("\n", "").split("\t")
				if index == 0:
					line[1] = [eval(i) for i in line[1]]
					weight_data.append(line)
				else:
					line[1] = [float(i) for i in line[1]]
					bias_data.append(line)
	return weight_data, bias_data

def normalized(dist):
	dist_sum = dist.sum()
	normed_dist = dist / dist_sum
	return normed_dist

def build_distributions(n, normalize=False, output=False):
	weight_data, bias_data = load_output_data()
	model_names = ["Zero-weights", "Dense", "Split"]
	# weight_dists {track models
	#					{track output node
	#						{track high and low weights separately
	#							{make distributions on those weights
	weight_dists = {model_name: {output_node: {pos_neg: np.zeros(hidden_size, dtype=int) for pos_neg in ["top", "low"]} for output_node in range(len(weight_data[0][1]))} for model_name in model_names}
	# bias_dists {track models
	#				{track output node
	#					{record how many times it was positive or negative [neg, pos]
	bias_dists = {model_name: {node: np.zeros(2, dtype=int) for node in range(2)} for model_name in model_names}

	for sample in weight_data:
		model_name, hidden_to_outer = sample[0], sample[1]
		for i, weights in enumerate(hidden_to_outer):
				weights = dict(enumerate(weights))
				sorted_weights = sorted(weights.items(), key=itemgetter(1), reverse=True)
				top_weights = list(list(zip(*sorted_weights[:n]))[0])
				low_weights = list(list(zip(*sorted_weights[-n:]))[0])
				weight_dists[model_name][i]["top"][top_weights] += 1
				weight_dists[model_name][i]["low"][low_weights] += 1

	for sample in bias_data:
		model_name, hidden_to_outer = sample[0], sample[1]
		for i, bias_val in enumerate(hidden_to_outer):
			if bias_val > 0:
				bias_dists[model_name][i][1] += 1
			else:
				bias_dists[model_name][i][0] += 1

	if normalize:
		norm_bias = deepcopy(bias_dists)
		norm_weight = deepcopy(weight_dists)
		for model_name in model_names:
			for node in range(len(weight_dists["Split"])):
				norm_bias[model_name][node] = normalized(norm_bias[model_name][node])
				for pos_neg in ["top", "low"]:
					norm_weight[model_name][node][pos_neg] = normalized(norm_weight[model_name][node][pos_neg])
		if output:
			pprint(norm_weight, width=1)
		return norm_weight, norm_bias
	if output:
		pprint(weight_dists, width=1)
	return weight_dists, bias_dists

def draw_distributions(n, normalize=False):
	weight_dists, bias_dists = build_distributions(n, normalize)
	name = {True: "Normalized", False: "Unnormalized"}
	draw_graphs(weight_dists, name[normalize], "Weights", n, which="both")
	draw_graphs(bias_dists, name[normalize], "Biases", n, which="both")
	draw_graphs(weight_dists, name[normalize], "Weights", n, which="top")
	draw_graphs(weight_dists, name[normalize], "Weights", n, which="low")

def draw_graphs(dists, name, datastyle, n, which="both"):
	model_names = ["Zero-weights", "Dense", "Split"] #column titles
	node_names = ["Output node " + str(i) for i in range(len(dists["Split"]))] #row titles
	labels = {"top": "Highest", "low": "Lowest", 1:"Positive", 0: "Negative"}
	key = {"Weights": ("top", "low"), "Biases": (1, 0)}
	colors = {"top": "g", "low": "r", 1:"g", 0: "r"}

	fig, axs = plt.subplots(len(node_names), 3, figsize=(12, 8), sharey=True, sharex=True)
	fig.suptitle(name + " Distribution of Positive and Negative " + datastyle + " for each output node")
	
	# Annotating row and columns
	pad = 5
	cols = axs
	rows = [axs[0]]
	if len(axs.shape) > 1:
		cols = axs[0]
		rows = axs[:,0]
	for ax, col in zip(cols, model_names):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
			xycoords="axes fraction", textcoords="offset points", 
			size="large", ha="center", va="baseline")
	for ax, row in zip(rows, node_names):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
				xycoords=ax.yaxis.label, textcoords="offset points",
				size="large", ha="right", va="center")

	fig.subplots_adjust(left=0.15, top=0.90)
	if len(axs.shape) == 1:
		axs = [axs]
	for node, row in enumerate(axs):
		for index, ax in enumerate(row):
			ind = np.arange(2)
			if datastyle == "Weights":
				ind = np.arange(50)
			if which == "both":
				width = 0.35
				key_1, key_0 = key[datastyle]
				top = dists[model_names[index]][node][key_1]
				low = dists[model_names[index]][node][key_0]
				bar1 = ax.bar(ind + width/2, top, width, label=labels[key_1], color="g")
				bar2 = ax.bar(ind - width/2, low, width, label=labels[key_0], color="r") 
			else:
				width=0.8
				vals = dists[model_names[index]][node][which]
				bar = ax.bar(ind, vals, width, label=labels[which], color=colors[which])
	axs[0][0].legend()
	plt.savefig(image_path + datastyle+ "_" + str(which) + "_" + str(n) + ".pdf")


def print_percentages(names=["Zero-weights", "Dense", "Split"]):
	# info with list of [sensitivity, specificity, correctness]
	percents = {"Zero-weights": [0]*3, "Dense": [0]*3, "Split": [0]*3}
	total = 0
	with open(percent_save_loc, "r") as f:
		for line in f:
			total += 1
			line = line.split()
			run_name = line[0]
			run_data = list(map(float, line[1:]))
			percents[run_name] = [percents[run_name][i] + run_data[i] for i in range(3)]

	total = total / 3
	print(" " * 11, "Sensitivity", "Specificity", "Correctness", sep="\t")
	for model in names:
		percents[model] = ["{0:.9f}".format(percents[model][i] / total) for i in range(3)]
		print("{0: <14}".format(str(model)), *percents[model], sep="\t")
		
# Check how often each hallmark node is positive and negative for the output
def weight_statistics(cutoff=0.3):
	weight_data, bias_data = load_output_data()
	model_names = ["Zero-weights", "Dense", "Split"]
	# {track each model
	#	{track each output node
	#		{track each hidden node
	#			track how many times that hidden node was [neg, pos, <-cutoff, >cutoff, neg_weight_sum, pos_weight_sum]
	node_stats = {
			model_name: {
				output_index:{
					hidden_index:np.zeros(8, dtype=float) 
					for hidden_index in range(len(weight_data[0][1][0]))} 
				for output_index in range(len(weight_data[0][1]))} 
			for model_name in model_names}

	for iteration in weight_data:
		model_name, node_data = iteration[0], iteration[1]
		for output_index, hidden_node_data in enumerate(node_data):
			for hidden_index, hidden_node in enumerate(hidden_node_data):
				current_data = node_stats[model_name][output_index][hidden_index]
				if hidden_node > 0:
					current_data[1] += 1
					current_data[5] += hidden_node
					if hidden_node > cutoff:
						current_data[3] += 1
						current_data[7] += hidden_node
				elif hidden_node < 0:
					current_data[0] += 1
					current_data[4] += hidden_node
					if hidden_node < -cutoff:
						current_data[2] += 1
						current_data[6] += hidden_node

	#print("Split | neg | pos | <", "-" + str(cutoff),  "| >", cutoff)
	#pprint(node_stats["Split"])
	#print("Zero-weights | neg | pos | <", "-" + str(cutoff), "| >", cutoff )
	#pprint(node_stats["Zero-weights"])

	special_nodes = {i:[] for i in ["Split"]} #["Zero-weights", "Split"]}
	for model_name in ["Split"]: #["Zero-weights", "Split"]:
		for output_index in [0]:
			for hidden_index in range(50):
				# node_data = [neg, pos, < neg cutoff, > cutoff]
				node_data = node_stats[model_name][output_index][hidden_index]
				crit_a = node_data[1] / node_data[0]
				crit_b = node_data[3] / node_data[1]
				crit_c = node_data[2] 
				pos_weight_avg = "{:.3f}".format(node_data[5] / node_data[1])
				neg_weight_avg = "{:.3f}".format(node_data[4] / node_data[0])
				pos_c_avg = "{:.3f}".format(node_data[7] / node_data[3])
				neg_c_avg = "{:.3f}".format(node_data[6] / node_data[2])
				if neg_c_avg == "nan": neg_c_avg = "-0.000" 
				# avg values are 2.2569, 0.833, 9.8
				print(model_name, "{0:2}".format(hidden_index), "{:.3f}".format(crit_a), "{:.3f}".format(crit_b), "{:.3f}".format(crit_c))
				# used before were 1.50, 0.75, and 5 but since I recollected data, numbers are different
				if crit_a >= 2.25 and crit_b >= 0.80 and crit_c <= 10:
					special_nodes[model_name].append(hidden_index)
					'''
					print("{0:12}".format(model_name), "{0:2}".format(hidden_index), 
							node_data[:-4].astype(int), 
							neg_weight_avg, pos_weight_avg,
							neg_c_avg, pos_c_avg)
					'''
					pass
	print("\n")
	for x in special_nodes:
		z = special_nodes[x]
		print(x)
		with open(text_gene_groups, "r") as f:
			for i, line in enumerate(f):
				node_data = node_stats[x][0][i]
				pos_c_avg = "{:.3f}".format(node_data[7] / node_data[3])
				neg_c_avg = "{:.3f}".format(node_data[6] / node_data[2])
				name = line.split()[0]
				if i in z:
					print("{0:2}".format(i), "{:40}".format(name), neg_c_avg, pos_c_avg)
					pass
		print(len(z))

def important_weights(n):
	weight_dists, bias_dists = build_distributions(5, normalize=False)
	split_dist_top = weight_dists["Split"][0]["top"].tolist()
	split_dist_low = weight_dists["Split"][0]["low"].tolist()
	sorted_top = sorted(enumerate(split_dist_top), key=itemgetter(1))[-n:]
	sorted_top = list(list(zip(*sorted_top))[0])
	
	sorted_low = sorted(enumerate(split_dist_low), key=itemgetter(1))[-n:]
	sorted_low = list(list(zip(*sorted_low))[0])

	print("top", sorted_top)
	print("low", sorted_low)

if __name__ == "__main__":
	#build_distributions(5, normalize=False, output=False)
	#draw_distributions(5, normalize=True)
	weight_statistics(cutoff=0.15)
	#important_weights(5)
	pass




