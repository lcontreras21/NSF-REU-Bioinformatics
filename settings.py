# Store settings, parameters, and files to use here so 
# don't have to change multiple things in the multiple files

data = "subset" # either 'full' or 'subset'
debug = False # set to True to display progress and other information
record_data = True # set to True to store model data and percentages in text files
test_behavior = False # set to True if some weight nodes need to be removed for experioments
seed = True # set to True if all instantiations should have the same weight matrixes
testing_parameters = False # set to True if the collected data should be erased after each session, helpful if testing different parameters and don't want overaccumulation 

if test_behavior:
	weights_to_test = [26, 27, 19, 34, 25, 3, 8, 13, 41]
else:
	weights_to_test = []


# hyperparameters
input_size, hidden_size, output_size = 4383, 50, 1
num_epochs = 1 
learning_rate = 0.001

# text files to use for the data
train_dir = "text_files/training_data/"
test_dir = "text_files/testing_data/"

# gene group data to use
text_gene_groups = "text_files/h.all.v6.2.symbols.txt"
#text_gene_groups = "text_files/c2.cp.kegg.v6.2.symbols.txt"
#hidden_size = 186 

stored_dict_locs = {name:"state_dicts/nn_" + name + ".pt" for name in ["Split", "Dense", "Zero-weights"]}

# where to save run save and store various files if enabled
text_path = "text_files/analysis/final/"
image_path = "diagrams/final/"

percent_save_loc = text_path + "percentages.txt"
fc1_weight_data_loc = text_path + "fc1_weights.txt"
fc1_bias_vals_loc = text_path + "fc1_bias_vals.txt"
fc1_gene_weights_loc = text_path + "fc1_gene_weights.pickle"

fc2_weight_data_loc = text_path + "fc2_weights.txt"
fc2_bias_data_loc = text_path + "fc2_bias.txt"

# other storage locations
starting_seed_loc = "text_files/starting_seed.pickle"
gene_pairs_loc = "text_files/gene_pairs.pickle"
