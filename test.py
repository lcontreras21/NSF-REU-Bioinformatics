from nn_1 import NN
import torch
import torch.optim as optim
'''
model = NN(2, 50, 20629)
model.load_state_dict(torch.load("state_dicts\\nn_1_state_dict.pt"))
optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer.load_state_dict(torch.load("state_dicts\\optim.pt"))
import pprint
pp = pprint.PrettyPrinter(indent=4)
state_dict = model.state_dict()

unaltered = state_dict[list(state_dict.keys())[0]]
print(unaltered[49, 7076])
print(unaltered[49, 6298])





for state_no in state_dict["state"]:
	state_info = state_dict["state"][state_no]
	for item in state_info:
		if type(state_info[item]) != type(20):
			print(state_info[item].size())



f = open("text_files\\h.all.v6.2.symbols.txt", "r")
gene_list = []
for line in f:
	gene_data = line.split()[2:]
	for gene in gene_data:
		if gene not in gene_list:
			gene_list.append(gene)

f.close()
print(len(gene_list))
'''

from nn_partial_links import *

g = open("text_files\\missing_genes.txt", "w")

#testing to see if all genes in hallmark are in cancer group
gene_data = gene_dict()
#print(gene_data["Zfp-112"])

gene_groups = import_gene_groups()
count = 0
for gene_group in gene_groups:
	for gene in gene_group:
		try:
			gene_data[gene]
		except:
			g.write(gene)
			g.write("\t")
			print(gene)
			count += 1
g.close()
print(count)
