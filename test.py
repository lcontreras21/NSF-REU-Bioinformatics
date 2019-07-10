import torch
import torch.optim as optim
import pickle
import numpy as np
from nn_partial_links import * 
'''
# find missing genes
f = open("text_files/missing_genes.txt", "r")
missing_genes = f.readline()
f.close()

gene_indexer = gene_dict()
gene_groups = import_gene_groups()

count = 0
for gene_group in gene_groups:
	for gene in gene_group:
		try:
			gene_indexer[gene]
		except:
			#print(gene)
			missing_genes += gene + "\t"
			count += 1
			pass
f = open("text_files/missing_genes.txt", "w")
f.write(missing_genes)
f.close()
print(count)

import collections
f = open("text_files/gene_pairs.pickle", "rb")
names = pickle.load(f)
f.close()

count = 0 
for key, value in names.items():
	if "/Search/" in value or "Check manually" == value:
		print(key, value)
	if "LOC" in key:
		count += 1
print(count)

# seeing if they are in gene database
g = open("text_files/first_line.txt", "r")
genes = g.readline().split()
g.close()

print(len(genes))
count = 0
for key, value in names.items():
	if value not in genes:
		print(key, value)
		count += 1
print(count)

'''
'''
# find missing genes in kegg

f = open("text_files/c2.cp.kegg.v6.2.symbols.txt", "r")
g = open("text_files/full_gene_names.txt", "r")
j = open("text_files/h.all.v6.2.symbols.txt", "r")
groups = []
for line in f:
	groups.append(line.split()[2:])

for line in j:
	groups.append(line.split()[2:])
x = g.readline().split()

f.close()
g.close()
j.close()

count = 0
missing_genes = []
for group in groups:
	for gene in group:
		if gene not in x and gene not in missing_genes:
			missing_genes.append(gene)
			count += 1
#k = open("text_files/missing_genes_kegg.txt", "w")
#k.write("\t".join(missing_genes))
print(count)
print(missing_genes)
#k.close()
'''
