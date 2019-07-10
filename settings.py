# Store settings, parameters, and files to use here so 
# Don't have to change multiple things in the model files

# data is either full, or subset
# mode is either train, test, or free
#data = "full"
data = "subset"
mode = "train"
debug = False 
record_data = True

test_behavior = True
weights_to_test = [26, 27, 19, 34, 25]

# testing parameters
if mode == "free":
	tumor_data_size = 200 
	normal_data_size = 30 
elif mode == "train":
	tumor_data_size = 7975
	normal_data_size = 584

samples_per_trial = 50

hidden_size = 50 

# hyperparameters
input_size = 35728
output_size = 2
num_epochs = 1 
learning_rate = 0.001

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
	text_data = logged

elif data == "subset" and mode == "train":
	text_file_normal =  train_dir + "subset_train_normal_samples.txt"
	text_file_tumor = train_dir + "subset_train_tumor_samples.txt"
	text_data = sub_logged
	input_size = 4579

elif data == "full" and mode == "train":
	text_file_normal = train_dir + "full_train_normal_samples.txt"
	text_file_tumor = train_dir + "full_train_tumor_samples.txt"
	text_data = logged

# gene group data to use
text_gene_groups = "text_files/h.all.v6.2.symbols.txt"
#text_gene_groups = "text_files/c2.cp.kegg.v6.2.symbols.txt"
#hidden_size = 186 


split_dict = "state_dicts/nn_split_test.pt"
dense_dict = "state_dicts/nn_dense_test.pt"
partial_dict = "state_dicts/nn_partial_test.pt"

# where to save run percentages if enabled
folder_path = "text_files/analysis/"
percent_save_loc = folder_path + "percentages_modded.txt"

# where to save weights and biases if enabled
bs_save_loc = folder_path + "biases_similar_modded.txt"
b_save_loc = folder_path + "biases_modded.txt"

ws_save_loc = folder_path + "weights_similar_modded.txt"
w_save_loc = folder_path + "weights_modded.txt"
