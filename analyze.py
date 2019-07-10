'''
Open the saved data from the files and analyze them
Check if numbers are repeated and how many times there are overlap
'''
from settings import *
from copy import deepcopy
from collections import Counter

def process_files():
	files = [ws_save_loc, w_save_loc, bs_save_loc, b_save_loc]
	to_return = []
	for name in files:
		f = open(name, "r")
		dist_data = []
		for line in f:
			data = line.split()
			dist_data.append((data[0], list(map(int,data[1:]))))
		f.close()
		to_return.append(dist_data)
	return to_return

def normalize(d):
	total = sum(d.values())
	d = {i:(d[i]/total) for i in d}
	return d

def process(data, data_type, normalized=False):
	# data = [w_s, w, b_s, b]
	# dists = [split, dense, partial, d-p, d-s, p-s]	
	dists = [{i:0 for i in range(hidden_size)} for i in range(6)]
	if data_type == "weights":
		x = data[0] + data[1]
	elif data_type == "biases":
		x = data[2] + data[3]
	for data_sample in x:
		name = data_sample[0]
		vals = data_sample[1]

		if name == "split":
			for i in vals:
				dists[0][i] += 1

		elif name == "dense":
			for i in vals:
				dists[1][i] += 1

		elif name == "partial":
			for i in vals:
				dists[2][i] += 1

		elif name == "dense-partial":
			for i in vals:
				dists[3][i] += 1

		elif name == "dense-split":
			for i in vals:
				dists[4][i] += 1

		elif name == "partial-split":
			for i in vals:
				dists[5][i] += 1
		else:
			raise ValueError("Something went wrong in the files here for the analysis.")

	if normalized:
		for i, dist in enumerate(dists):
			norm_dist = normalize(dist)
			dists[i] = norm_dist
	return dists

# Three types of data: dense, split, partial
import matplotlib.pyplot as plt
def draw_graph(dists, title, save_location="diagrams/distribution.pdf"):
	# dists = [split, dense, partial, d-p, d-s, p-s]	
	fig, axs = plt.subplots(3, 2, sharey=True) #,sharex=True, sharey=True)
	(ax0, ax1), (ax2, ax3), (ax4, ax5) = axs
	fig.suptitle("Distribution of " + title)
	fig.subplots_adjust(hspace=0.5)
	plt.rcParams['xtick.labelsize'] = 4


	ax0.bar(list(dists[0].keys()), list(dists[0].values()), color='r', align='center')
	#ax0.set_xticks(ticks=list(dists[0].keys()))
	ax0.set_xticklabels(labels=list(dists[0].keys()), minor=True, rotation='vertical')
	ax0.set_title("Split Model")
	
	ax1.bar(list(dists[1].keys()), list(dists[1].values()), color='g')
	ax1.set_title("Dense Model")
	
	ax2.bar(list(dists[2].keys()), list(dists[2].values()), color='b')
	ax2.set_title("Zero-weights Model")

	ax3.bar(list(dists[3].keys()), list(dists[3].values()), color='tab:grey')
	ax3.set_title("Dense-Zero Overlap")
	
	ax4.bar(list(dists[4].keys()), list(dists[4].values()), color='tab:brown')
	ax4.set_title("Dense-Split Overlap")
	
	ax5.bar(list(dists[5].keys()), list(dists[5].values()), color='tab:purple')
	ax5.set_title("Zero-Split Overlap")
	plt.savefig(save_location)

def draw_graphs(which="both"):
	# which can be one of {'both', 'weights', biases'}
	processed_data = process_files()
	if which == "weights" or which == "both":
		dists_unnorm = process(processed_data, "weights", normalized=False)
		draw_graph(dists_unnorm,  "Top 5 Weights", save_location="diagrams/w_unnormalized_dist" + modded + ".pdf")

		dists_norm = process(processed_data, "weights", normalized=True)
		draw_graph(dists_norm,  "Top 5 Weights", save_location="diagrams/w_normalized_dist" + modded + ".pdf")
	elif which == "biases" or which == "both":
		dists_unnorm = process(processed_data, "biases", normalized=False)
		draw_graph(dists_unnorm,  "Top 5 Biases", save_location="diagrams/b_unnormalized_dist" + modded + ".pdf")

		dists_norm = process(processed_data, "biases", normalized=True)
		draw_graph(dists_norm,  "Top 5 Biases", save_location="diagrams/b_normalized_dist" + modded + ".pdf")

def biggest_weights(n, pretty_print=False):
	# Necessary stuff to be able to get info from any file
	processed_data = process_files()
	dists = process(processed_data, "weights", normalized=False)
	
	# returning top five biggest weights
	top = []
	for dist in dists:
		top.append(dict(Counter(dist).most_common(n)))
	for i, item in enumerate(top):
		x = sorted(item.items(), key=lambda x: x[1], reverse=True)
		output = "{"
		for pair in x:
			output += "{:02}".format(pair[0]) + ": " + "{:02}".format(pair[1]) + ", "
		output += "\b\b}"
		if pretty_print:
			print(output)
		top[i] = x
	# top = [split, dense, zerow, dz, ds, zs] dicts 
	return top[:n]

def closeness():
	processed_dists = biggest_weights(50)
	split = processed_dists[0]
	zerow = processed_dists[2]
	
	same = []
	close = []

	for i, (key, value) in enumerate(split):
		try:
			if key == zerow[i - 1][0]:
				close.append((key, value, zerow[i - 1][1]))
			elif key == zerow[i + 1][0]:
				close.append((key, value, zerow[i + 1][1]))
		except:
			continue
		if key == zerow[i][0]:
			same.append((key, value, zerow[i][1]))
	print("If the split and the zero-weights models had the same weights at the same index")
	print(same)
	print("If the split and the zero-weights models have a similar weight index")
	print(close)
	print("How many times the two models overlapped during the same session")
	print(processed_dists[5][:5])

def print_percentages():
	f = open(percent_save_loc, "r")
	# info with list of [sensitivity, specificity, correctness]
	zerow = [0, 0, 0]
	dense = [0, 0, 0]
	split = [0, 0, 0]
	total = 0
	for line in f:
		total += 1
		data = line.split()
		name = data[0]
		data = list(map(float, data[1:]))
		if name == "Zero-weights":
			zerow[0] += data[0]
			zerow[1] += data[1]
			zerow[2] += data[2]
		elif name == "Dense":
			dense[0] += data[0]
			dense[1] += data[1]
			dense[2] += data[2]
		elif name == "Split":
			split[0] += data[0]
			split[1] += data[1]
			split[2] += data[2]
	total = total / 3
	zerow = ["{0:.4f}".format(zerow[i] / total) for i in range(len(zerow))]
	dense = ["{0:.4f}".format(dense[i] / total) for i in range(len(dense))]
	split = ["{0:.4f}".format(split[i] / total) for i in range(len(split))]

	print("Zerow", *zerow, sep=", ")
	print("Dense", *dense, sep=", ")
	print("Split", *split, sep=", ")

def verify_removed_weights():
	if test_behavior: 
		processed_data = process_files()
		# only interested in the main files, not overlap
		# that data is at first and third index
		data = [processed_data[1], processed_data[3]]
		name = ["weights", "biases"]

		# check if any of the weights_to_test are in there
		count = 0
		for i, dataset in enumerate(data):
			for ii, line in enumerate(dataset):
				model_name, big_weights = line
				error = name[i] + " " + model_name + " " + str(ii) + "\n"
				mistakes = [weight for weight in weights_to_test if weight in big_weights and model_name != "dense"]
				if len(mistakes) != 0:	
					print(error.join([str(x) for x in mistakes]))
				count += len(mistakes)
		if count == 0:
			print("Weights were properly removed")
	else:
		print("No need to test important weights. Make sure this is intentional.")

if __name__ == "__main__":
	draw_graphs(which="weights")
	#closeness()
	#verify_removed_weights()
	#print_percentages()
