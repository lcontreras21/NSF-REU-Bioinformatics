import random

# Make a subset dataset based on the gene groups
# Save that alternative to /text_files/
def make_subset_dataset():
	gene_groups = import_gene_groups()
	gene_groups_combined = [gene for gene_group in gene_groups for gene in gene_group]
	#get unique genes in gene_groups
	genes = list(set(gene_groups_combined))
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

# Make tumor and normal data sets based on the subset
def norm_tum_sets():
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

# Make name files to be used in later files
def name_files():
	y = open("text_files/subset_logged_scaled_rnaseq.txt", "r")
	u = open("text_files/subset_gene_names.txt", "w")

	data = y.readline()
	u.write(data)

	y.close()
	u.close()

def create_train_test_data(name):
	files = {"_normal_samples.txt":(584, 146), "_tumor_samples.txt": (7975, 1994)}
	for f in files:
		training_size, testing_size = files[f]
		all_samples = []
		with open("text_files/" + name + f) as text_file:
			for line in text_file:
				all_samples.append(line)
		random.shuffle(all_samples)
		training = all_samples[:training_size]
		testing = all_samples[-testing_size:]
		
		text_files = {"train": ("text_files/training_data/", training),"test": ("text_files/testing_data/", testing)}
		for t in text_files:
			with open(text_files[t][0] + name + text_file + f, "w") as datafile:
				for line in text_files[t][1]:
					datafile.write(line)

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
