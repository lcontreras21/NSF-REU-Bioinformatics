# Store settings, parameters, and files to use here so 
# Don't have to change multiple things in the model files

# six modes: use_subset, use_full, train_subset, test_subset, train_full, test_full

mode = "train_full"

# testing parameters

### TODO: fix data sizes below to reflect training and testing
if mode == "use_subset" or mode == "use_full":
	tumor_data_size = 200 
	normal_data_size = 20
elif mode == "train_subset" or mode == "train_full":
	tumor_data_size = 7975
	normal_data_size = 584
elif mode == "test_subset" or mode == "test_full":
	tumor_data_size = 1994
	normal_data_size = 146

samples_per_trial = 50
trials = 1
hidden_size = 50

# hyperparameters
input_size = 35728
output_size = 2
num_epochs = 3
learning_rate = 0.01

# text files to use for the data
if mode == "use_subset":
	text_file_normal = "text_files/subset_normal_samples.txt"
	text_file_tumor = "text_files/subset_tumor_samples.txt"
	text_data = "text_files/subset_logged_scaled_rnaseq.txt"
	input_size = 4579
elif mode == "use_full":
	text_file_normal = "text_files/full_normal_samples.txt"
	text_file_tumor = "text_files/full_tumor_samples.txt"
	text_data = "text_files/logged_scaled_rnaseq.txt"

elif mode == "train_subset":
	text_file_normal = "text_files/training_data/subset_train_normal_samples.txt"
	text_file_tumor = "text_files/training_data/subset_train_tumor_samples.txt"
	text_data = "text_files/subset_logged_scaled_rnaseq.txt"
	input_size = 4579
elif mode == "test_subset":
	text_file_normal = "text_files/testing_data/subset_test_normal_samples.txt"
	text_file_tumor = "text_files/testing_data/subset_test_tumor_samples.txt"
	text_data = "text_files/subset_logged_scaled_rnaseq.txt"
	input_size = 4579

elif mode == "train_full":
	text_file_normal = "text_files/training_data/full_train_normal_samples.txt"
	text_file_tumor = "text_files/training_data/full_train_tumor_samples.txt"
	text_data = "text_files/logged_scaled_rnaseq.txt"
elif mode == "test_full":
	text_file_normal = "text_files/testing_data/full_test_normal_samples.txt"
	text_file_tumor = "text_files/testing_data/full_test_tumor_samples.txt"
	text_data = "text_files/logged_scaled_rnaseq.txt"


# gene group data to use
text_gene_groups = "text_files/h.all.v6.2.symbols.txt"
#text_gene_groups = "text_files/c2.cp.kegg.v6.2.symbols.txt"




