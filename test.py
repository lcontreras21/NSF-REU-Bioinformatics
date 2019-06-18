import torch
import torch.optim as optim
import pickle
import numpy as np
from nn_partial_links import * 
'''
hallmark = open("text_files/h.all.v6.2.symbols.txt", "r")
kegg = open("text_files/c2.cp.kegg.v6.2.symbols.txt","r")

h_count = 0

for line in hallmark:
	current_line = line.split()
	h_count += len(current_line[2:])

k_count = 0
for line in kegg:
	current_line = line.split()
	k_count += len(current_line[2:])

print(h_count, k_count)
hallmark.close()
kegg.close()

# find missing genes in hallmark
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

'''
'''
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
# making alternate data set with only the genes in hallmark
gene_groups = import_gene_groups()

#get unique genes in gene_groups
genes = []
for gene_group in gene_groups:
	for gene in gene_group:
		if gene not in genes:
			genes.append(gene)

# importing dict of indices for gene in rnaseq dataset			
genes_dict = gene_dict()

#getting indices from unique genes, it is sorted
gene_indices = get_gene_indicies(genes, genes_dict)
gene_indices = [0] + gene_indices + [len(gene_groups[0]) - 1]
f = open("text_files/logged_scaled_rnaseq.txt", "r")
# open new file for subset of data
g = open("text_files/subset_logged_scaled_rnaseq.txt", "w")
index = 0
for line in f:
	if index % 100 == 0:
		print(index)
	index += 1
	data = line.split()
	# remove entries from list that aren't necessary
	new_data = []
	prev = 0
	for item in gene_indices:
		data[prev: item] = [0] * (item - prev)
		prev = item + 1
	new_data = list(filter((0).__ne__, data)) 
	to_write = "\t".join(new_data)
	g.write(to_write+"\n")
	
print(len(new_data))
f.close()
g.close()
'''


'''
# create files for tumor data and normal data based on subset
#### TODO preserver tumor/normal information in the above chunk of code
f = open("text_files/subset_logged_scaled_rnaseq.txt", "r")
g = open("text_files/subset_normal_samples.txt", "w")
h = open("text_files/subset_tumor_samples.txt", "w")

tumor = 0
normal = 0
for line in f:
	data = line.split()
	if data[-1] == "Tumor":
		h.write(line)
		tumor += 1
	if data[-1] == "Normal":
		g.write(line)
		normal += 1
print(tumor, normal)

f.close()
g.close()
h.close()
'''

'''
# keeping track of gene names in the subset data
f = open("text_files/subset_logged_scaled_rnaseq.txt", "r")
g = open("text_files/subset_gene_names.txt", "w")

data = f.readline()
g.write(data)

f.close()
g.close()
'''
'''
# making training data for full and subset to be used for 80 - 20
f = open("text_files/full_normal_samples.txt", "r")
g = open("text_files/full_tumor_samples.txt", "r")

h = open("text_files/full_train_normal_samples.txt", "w")
i = open("text_files/full_train_tumor_samples.txt", "w")
j = open("text_files/full_test_normal_samples.txt", "w")
k = open("text_files/full_test_tumor_samples.txt", "w")

index = 0
for x in range(7975): # train tumor set
	line = g.readline()
	i.write(line)
	index +=1 
print(index)
index = 0
for x in range(1994): # test tumor set
	line = g.readline()
	k.write(line)
	index +=1 
print(index)
index = 0
for x in range(584): # train normal set
	line = f.readline()
	h.write(line)
	index +=1 
print(index)
index = 0
for x in range(146): # test normal set
	line = f.readline()
	j.write(line)
	index +=1 
print(index)
index = 0

f.close()
g.close()
h.close()
i.close()
j.close()
k.close()


'''


'''

#make subset train and test dataset
f = open("text_files/subset_normal_samples.txt", "r")
g = open("text_files/subset_tumor_samples.txt", "r")

h = open("text_files/subset_train_normal_samples.txt", "w")
i = open("text_files/subset_train_tumor_samples.txt", "w")
j = open("text_files/subset_test_normal_samples.txt", "w")
k = open("text_files/subset_test_tumor_samples.txt", "w")

index = 0
for x in range(7975): # train tumor set
	line = g.readline()
	i.write(line)
	index +=1 
print(index)
index = 0
for x in range(1994): # test tumor set
	line = g.readline()
	k.write(line)
	index +=1 
print(index)
index = 0
for x in range(584): # train normal set
	line = f.readline()
	h.write(line)
	index +=1 
print(index)
index = 0
for x in range(146): # test normal set
	line = f.readline()
	j.write(line)
	index +=1 
print(index)
index = 0

f.close()
g.close()
h.close()
i.close()
j.close()
k.close()
'''




