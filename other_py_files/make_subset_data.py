import random
import os
from models.process_data import *

# Make a subset dataset based on the gene groups
# Save that alternative to /text_files/
def make_subset_dataset():
	gene_groups = import_gene_groups()
	gene_groups_combined = [gene for gene_group in gene_groups for gene in gene_group]
	#get unique genes in gene_groups
	genes = list(set(gene_groups_combined))

	# importing dict of indices for gene in full rnaseq dataset
	genes_dict = gene_dict("full")

	#getting indices from unique genes, it is sorted
	gene_indices = get_gene_indicies(genes, genes_dict)

	f = open("text_files/logged_scaled_rnaseq.txt", "r")

	# open new file for subset of data
	g = open("text_files/subset_logged_scaled_rnaseq.txt", "w")
	for line in f:
		sample = line.split()
		data_used = sample[1:-1]
		subset_data = [data_used[index] for index in gene_indices]
		to_write = "\t".join([sample[0]] + subset_data + [sample[-1]])
		g.write(to_write+"\n")

	f.close()
	g.close()

# Make tumor and normal data sets based on the subset
def norm_tum_sets():
	for file_set in ["", "subset_"]:

		f = open("text_files/" + file_set + "logged_scaled_rnaseq.txt", "r")
	
		if file_set == "":
			file_set = "full_"
	
		g = open("text_files/" + file_set + "normal_samples.txt", "w")
		h = open("text_files/" + file_set + "tumor_samples.txt", "w")

		for line in f:
			data = line.split()
			if data[-1] == "Tumor":
				h.write(line)
			if data[-1] == "Normal":
				g.write(line)
		f.close()
		g.close()
		h.close()

# to make train and test data, the samples already have to be split into normal and tumor
def create_train_test_data(data=data):
	
	for i in ["training_data/", "testing_data/", "analysis/"]:
		if not os.path.exists("text_files/" + i):
			os.mkdir("text_files/" + i)

	files = {"_normal_samples.txt":(584, 146), "_tumor_samples.txt": (7975, 1994)}
	for f in files:
		training_size, testing_size = files[f]
		all_samples = []
		with open("text_files/" + data + f) as text_file:
			for line in text_file:
				all_samples.append(line)
		random.shuffle(all_samples)
		training = all_samples[:training_size]
		testing = all_samples[-testing_size:]
		
		text_files = {"_train": ("text_files/training_data/", training),"_test": ("text_files/testing_data/", testing)}
		for text_file in text_files: 
			loc	= text_files[text_file][0] + data + text_file + f
			# example: "text_files/training_data/full_train_normal_samples.txt"
			with open(loc, "w") as train_test_file:
				for line in text_files[text_file][1]:
					train_test_file.write(line)

# Make name files to be used in later files
def save_gene_names(file_i=""):
	y = open("text_files/" + file_i + "logged_scaled_rnaseq.txt", "r")
	if file_i == "":
		file_i = "full_"
	u = open("text_files/" + file_i + "gene_names.txt", "w")

	all_gene_names = y.readline()
	u.write(all_gene_names)

	y.close()
	u.close()


# bottom function not needed unless another gene group is added
# there are only 4 missing genes in hallmark. much more in kegg
def find_missing_genes():
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
	print(missing_genes)
