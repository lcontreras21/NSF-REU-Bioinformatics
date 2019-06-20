# Store settings, parameters, and files to use here so 
# Don't have to change multiple things in the model files

# data is either full, or subset
# mode is either train, test, or free
data = "full"
mode = "test"

# testing parameters

if mode == "free":
	tumor_data_size = 50 
	normal_data_size = 50
elif data == "train":
	tumor_data_size = 7975
	normal_data_size = 584
elif mode == "test":
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
train_dir = "text_files/training_data/"
test_dir = "text_files/testing_data/"
sub_logged = "text_files/subset_logged_scaled_rnaseq.txt"
logged = "text_files/logged_scaled_rnaseq.txt"

if data == "subset" and mode == "free":
	text_file_normal = "text_files/subset_normal_samples.txt"
	text_file_tumor = "text_files/subset_tumor_samples.txt"
	text_data = sub_logged
	input_size = 4579
elif data == "full" and mode == "free":
	text_file_normal = "text_files/full_normal_samples.txt"
	text_file_tumor = "text_files/full_tumor_samples.txt"
	text_data = "text_files/logged_scaled_rnaseq.txt"

elif data == "subset" and mode == "train":
	text_file_normal =  train_dir + "subset_train_normal_samples.txt"
	text_file_tumor = train_dir + "subset_train_tumor_samples.txt"
	text_data = sub_logged
	input_size = 4579

elif data == "subset" and mode == "test":
	text_file_normal = test_dir + "subset_test_normal_samples.txt"
	text_file_tumor = test_dir + "subset_test_tumor_samples.txt"
	text_data = sub_logged
	input_size = 4579

elif data == "full" and mode == "train":
	text_file_normal = train_dir + "full_train_normal_samples.txt"
	text_file_tumor = train_dir + "full_train_tumor_samples.txt"
	text_data = logged

elif data == "full" and mode == "test":
	text_file_normal = test_dir + "full_test_normal_samples.txt"
	text_file_tumor = test_dir + "full_test_tumor_samples.txt"
	text_data = logged


# gene group data to use
text_gene_groups = "text_files/h.all.v6.2.symbols.txt"
#text_gene_groups = "text_files/c2.cp.kegg.v6.2.symbols.txt"



# models to test
from nn_partial_links import * 
from nn_split import NN_split

split_dict = "state_dicts/nn_split_full.pt"
dense_dict = "state_dicts/nn_dense_full.pt"
partial_dict = "state_dicts/nn_partial_full.pt"





