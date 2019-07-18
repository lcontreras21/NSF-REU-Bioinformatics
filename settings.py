# Store settings, parameters, and files to use here so 
# Don't have to change multiple things in the model files
# data is either full, or subset
# mode is train, changed to test in testing_models.py file

data = "subset"
mode = "train"
debug = False
record_data = True
test_behavior = False
seed = True

if test_behavior:
	weights_to_test = [26, 27, 19, 34, 25, 3, 8, 13]
else:
	weights_to_test = []


# hyperparameters
input_size, hidden_size, output_size = 35728, 50, 2
num_epochs = 1 
learning_rate = 0.001
tumor_data_size, normal_data_size = 7975, 584

# text files to use for the data
train_dir = "text_files/training_data/"
test_dir = "text_files/testing_data/"
sub_logged = "text_files/subset_logged_scaled_rnaseq.txt"
logged = "text_files/logged_scaled_rnaseq.txt"

if data == "subset" and mode == "train":
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

stored_dict_locs = {name:"state_dicts/nn_" + name + ".pt" for name in ["Split", "Dense", "Zero-weights"]}

# where to save run percentages if enabled
text_path = "text_files/analysis/fixed_seed/"
image_path = "diagrams/fixed_seed/"
modded = ""
replace = "_modded"
if test_behavior:
	modded = replace

percent_save_loc = text_path + "percentages" + modded + ".txt"
all_weight_data_loc = text_path + "all_weights.txt"

# where to save weights and biases if enabled
bs_save_loc = text_path + "biases_similar" + modded + ".txt"
b_save_loc = text_path + "biases" + modded + ".txt"

ws_save_loc = text_path + "weights_similar" + modded + ".txt"
w_save_loc = text_path + "weights" + modded + ".txt"

# other storage locations
starting_seed_loc = "text_files/starting_seed.pickle"
gene_pairs_loc = "text_files/gene_pairs.pickle"

