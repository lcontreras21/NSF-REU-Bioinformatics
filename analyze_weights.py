'''
Open the saved data from the files and analyze them
Check if numbers are repeated and how many times there are overlap
'''
from settings import hidden_size
from copy import deepcopy

def preprocess(combined=False):
	loc = "text_files/analysis/"
	files = ["weights_similar.txt", "weights.txt", "biases_similar.txt", "biases.txt"]
	to_return = []
	for name in files:
		f = open(loc + name, "r")
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

# Three types of data: dense, split, partial
import matplotlib.pyplot as plt
def draw_graphs(data, data_type, title, save_location="diagrams/distribution.pdf", normalized=False):
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

	# dists = [split, dense, partial, d-p, d-s, p-s]	
	fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
	(ax0, ax1), (ax2, ax3), (ax4, ax5) = axs
	fig.suptitle("Distribution of " + title)
	fig.subplots_adjust(hspace=0.5)

	ax0.bar(list(dists[0].keys()), list(dists[0].values()), color='r')
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


if __name__ == "__main__":
	processed_data = preprocess()
	draw_graphs(processed_data, "weights",  "Top 5 Weights", save_location="diagrams/w_unnormalized_dist.pdf")
	draw_graphs(processed_data, "weights", "Top 5 Weights Normalized", save_location="diagrams/w_normalized_dist.pdf", normalized=True)

	draw_graphs(processed_data, "biases",  "Top 5 Biases", save_location="diagrams/b_unnormalized_dist.pdf")
	draw_graphs(processed_data, "biases", "Top 5 Weights Normalized", save_location="diagrams/b_normalized_dist.pdf", normalized=True)
	
