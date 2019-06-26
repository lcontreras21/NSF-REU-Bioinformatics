'''
Plotting the neural networks to show gene importance
'''

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import torch 
import numpy as np
import time
from datetime import timedelta 
from collections import OrderedDict

def group_vertices(model):
	'''
	if model == dense
		return everything connected
	if model == split
		return weighted edge with 0
	if model == zero
		return wieghted edge with low probability
	'''
	vertices_input = [] 
	if model == "dense":
		for gene in gene_names:
			vertices_input += [(gene, i, 1) for i in range(len(gene_groups))]
	
	elif model == "zero":
		for gene in gene_names:
			no_connections = list(range(50))
			gene_locations = find_gene(gene, gene_groups)
			no_connections = [x for x in no_connections if x not in gene_locations]
			#vertices_input += [(gene, i, 0.25) for i in no_connections]
			vertices_input += [(gene, i, 1) for i in gene_locations]

	elif model == "split":
		for gene in gene_names:
			gene_locations = find_gene(gene, gene_groups)
			vertices_input += [(gene, i, 1) for i in gene_locations]
	
	
	vertices_output = []
	vertices_output += [(i, "Tumor", 1) for i in range(len(gene_groups))]
	vertices_output += [(i, "Normal", 1) for i in range(len(gene_groups))]
	return vertices_input, vertices_output


def position():
	pos = {}
	for i, gene in enumerate(gene_names):
		pos[gene] = np.array([-9.5, len(gene_names) - (2 * (i + 1))])
	for group in range(len(gene_groups)):
		scaling_ratio = len(gene_names) / len(gene_groups)
		pos[group] = np.array([0, len(gene_names) - (2 * scaling_ratio * (group + 1))])
	pos["Tumor"] = np.array([9, int(len(gene_names) * (2/3))])
	pos["Normal"] = np.array([9, -int(len(gene_names) * (2/3))])
	return pos

def network_graph(model):
	time_model = time.monotonic()
	
	print("\nCurrently working on:", model)
	G = nx.Graph()
	print("Collecting nodes", flush=True)	
	G.add_nodes_from(gene_names)
	G.add_nodes_from(range(len(gene_groups)))
	G.add_nodes_from(["Tumor", "Normal"])

	print("Collecting vertices", flush=True)
	vertices_input, vertices_output = group_vertices(model)
	G.add_weighted_edges_from(vertices_input + vertices_output)
	
	print("Fixing node and edge poisitons on graph", flush=True)
	pos = position()

	plt.figure(model, figsize=(10, 10))
	plt.title(model)
	plt.axis([-10, 10, -len(gene_names), len(gene_names)])
	plt.xlabel("Input layer        |    Hidden Layer    |        Output Layer")
	
	print("Drawing nodes", flush=True)
	if model == "split" or model == "zero":
		used_genes = [item[0] for item in vertices_input]
	else:
		used_genes = gene_names
	nx.draw_networkx_nodes(G,
		nodelist=used_genes,
		pos=pos,
		node_size=0.1,
		node_shape='>',
		alpha=0.25)
	nx.draw_networkx_nodes(G,
		nodelist=list(range(len(gene_groups))),
		pos=pos,
		node_size=25,
		node_shape='h',
		alpha=1)
	nx.draw_networkx_nodes(G,
		nodelist=["Tumor", "Normal"],
		pos=pos,
		node_size=300,
		node_shape='8',
		alpha=1)
	
	print("Drawing vertices in the input->hidden layer", flush=True)
	esmall = [(u, v) for (u, v, d) in vertices_input if d == 0.25]
	elarge = [(u, v) for (u, v, d) in vertices_input if d == 1]
	
	'''
	nx.draw_networkx_edges(G,
		pos=pos,
		edgelist=esmall,
		alpha=0.01,
		style="dotted",
		width=0.001)
	'''
	nx.draw_networkx_edges(G,
		pos=pos,
		edgelist=elarge,
		alpha=0.25,
		style="solid",
		width=0.01)
	print("Drawing vertices in the hidden->output layer", flush=True)
	nx.draw_networkx_edges(G,
		pos=pos,
		edgelist=vertices_output,
		width=0.5)
	'''
	nx.draw_networkx_labels(G,
		pos=pos,
		font_size=0.1,
		alpha=0.25)
	'''
	print("Saving to file","diagrams/" + model + ".pdf", flush=True)
	plt.savefig("diagrams/" + model + ".pdf")
	plt.close(model)
	print("Done", flush=True)
	print("Time to draw model:", timedelta(seconds=time.monotonic() - time_model), flush=True)

if __name__ == "__main__":
	start_time = time.monotonic()	

	#f = open(state_dict_mem, "rb")
	#state_dict = torch.load(f)
	#f.close()
	print("Doing some preprocessing", flush=True)
	gene_names_file = "text_files/full_gene_names.txt"
	gene_group_file = "text_files/h.all.v6.2.symbols.txt"

	g = open(gene_names_file, "r")
	gene_names = list(OrderedDict.fromkeys(g.readline().split()[1:-1]))
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
	
	network_graph("split")
	print()
	network_graph("zero")
	print()
	network_graph("dense")
	print()
	print("Time Total:", timedelta(seconds=time.monotonic() - start_time), flush=True)
	print()
