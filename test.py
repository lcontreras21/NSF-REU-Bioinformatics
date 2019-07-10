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
print(len(genes), flush=True)

# importing dict of indices for gene in rnaseq dataset			
genes_dict = gene_dict()

#getting indices from unique genes, it is sorted
gene_indices = get_gene_indicies(genes, genes_dict)
print(len(gene_indices), flush=True)
gene_indices = [0] + gene_indices + [len(gene_groups[0]) - 1]

f = open("text_files/logged_scaled_rnaseq.txt", "r")
# open new file for subset of data

g = open("text_files/subset_logged_scaled_rnaseq.txt", "w")
index = 0
for line in f:
	data = line.split()
	data_used = data[1:-1]
	subset_data = [data_used[index] for index in gene_indices]
	to_write = "\t".join([data[0]] + subset_data + [data[-1]])
	g.write(to_write+"\n")
	
print(len(subset_data))
f.close()
g.close()


'''

'''
# create files for tumor data and normal data based on subset
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
y = open("text_files/subset_logged_scaled_rnaseq.txt", "r")
u = open("text_files/subset_gene_names.txt", "w")

data = y.readline()
u.write(data)

y.close()
u.close()
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

'''
gene_file = open("text_files/gene_pairs.pickle", "rb")
x = pickle.load(gene_file)
gene_file.close()
f = open("text_files/missing_genes_unique.txt", "r")
y = f.readline().split()
f.close()


def gene_dict():
	f = open("text_files/logged_scaled_rnaseq.txt", "r")
	gene_names = f.readline().split()
	f.close()

	gene_to_index = {}
	for index, gene_name in enumerate(gene_names[1:-1]):
		if gene_name not in gene_to_index:
			gene_to_index[gene_name] = [index]
		else:
			gene_to_index[gene_name] = gene_to_index[gene_name] + [index]
	return gene_to_index
dicti = gene_dict()
missing_genes = []
count = 0
for gene in y:
	if x[gene] != "Check manually":
		gene = x[gene]
	try:
		dicti[gene]
	except:
		missing_genes.append(gene)
		count += 1
print(count)
#x = open("text_files/actual_missing_genes.txt", "w")
#x.write("\t".join(missing_genes))
#x.close()

