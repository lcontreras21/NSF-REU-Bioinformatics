'''
Plotting the neural networks to show gene importance
'''

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import networkx as nx
import torch 
import numpy as np
import time
from datetime import timedelta 

def group_vertices():
	'''
	if model == dense
		return everything connected
	if model == split
		return weighted edge with 0
	if model == zero
		return wieghted edge with low probability
	'''
	vertices = [] 
	if model == "dense":
		for gene in gene_names:
			vertices += [(gene, i, 1) for i in range(len(gene_groups))]
	
	elif model == "zero":
		for gene in gene_names:
			no_connections = list(range(50))
			gene_locations = find_gene(gene, gene_groups)
			no_connections = [x for x in no_connections if x not in gene_locations]

			vertices += [(gene, i, 0.25) for i in no_connections]
			vertices += [(gene, i, 1) for i in gene_locations]

	elif model == "split":
		for gene in gene_names:
			gene_locations = find_gene(gene, gene_groups)
			vertices += [(gene, i, 1) for i in gene_locations]

	vertices += [(i, "Tumor", 1) for i in range(len(gene_groups))]
	vertices += [(i, "Normal", 1) for i in range(len(gene_groups))]
	return vertices

def position():
	pos = {}
	location = 0.6
	change = 0.00003587
	for i, gene in enumerate(gene_names):
		pos[gene] = np.array([-1, location - (change * (i+1))])
	for group in range(len(gene_groups)):
		pos[group] = np.array([0, location - (0.024 * (group + 1))])
	pos["Tumor"] = np.array([1, 0.3])
	pos["Normal"] = np.array([1, -0.3])
	return pos
def network_graph():
	'''
	Return the network graph based on the state_dict stored 
	in memory.
	'''

	G = nx.Graph()
	print("Adding nodes", flush=True)	
	G.add_nodes_from(gene_names)
	G.add_nodes_from(range(len(gene_groups)))
	G.add_nodes_from(["Tumor", "Normal"])

	print("Adding weighted edges", flush=True)
	G.add_weighted_edges_from(group_vertices())
	
	print("Fixing positions on graph", flush=True)
	pos = position()

	print("Drawing graph and saving to file", flush=True)
	plt.figure(1)
	nx.draw(G, pos)
	plt.savefig("graph.png")

if __name__ == "__main__":
	start_time = time.monotonic()	
	model = "dense" # dense, split, zero

	#f = open(state_dict_mem, "rb")
	#state_dict = torch.load(f)
	#f.close()
	print("Doing some preprocessing", flush=True)
	gene_names_file = "text_files/full_gene_names.txt"
	gene_group_file = "text_files/h.all.v6.2.symbols.txt"

	g = open(gene_names_file, "r")
	gene_names = g.readline().split()
	g.close()

	h = open(gene_group_file, "r")
	gene_groups = []
	for line in h:
		gene_groups.append(line.split())
	h.close()

	def find_gene(gene, gene_groups):
		index = []
		for i, gene_group in enumerate(gene_groups):
			if gene in gene_group:
				index.append(i)
		return index
	
	network_graph()
	print("Time elapsed:", timedelta(seconds=time.monotonic() - start_time), flush=True)

