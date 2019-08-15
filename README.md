# Bioinformatics NSF REU Project

## Title and Description
###### Official Title: Development of Machine Learning Models for Cancer Driver Identification
Large-scale cancer genome sequencing projects have resulted in the discovery of numerous cancer-specific genetic changes. However, it remains unclear how many of these genetic changes contribute to tumorigenesis.

In this project, the goal will be to leverage large omics databases and existing biological knowledge to learn an interpretable machine learning model that can differentiate between cancer samples and normal samples based on these genetic changes. Unlike previous black box models, the goal is to build a model that will be interpretable and will enable mechanistic interpretation by explaining how specific genetic changes contribute to individual cancer processes. The student’s project will contribute to a larger effort for cancer drug target discovery.

Outcome: The student will learn how to build a machine learning model to solve a key problem in cancer research. Many of the techniques and tools learned during this project can be applied to other machine learning projects. Ultimately, the student’s work will lead to the development of an important tool for the research community.

## Contents
- Three models: a fully connected (dense) model; a partially connected (zero-weights) model; a sparsely connected (split) model. Main model being used is the split model with zero-weights as a sanity check and dense used as a benchmark. Each model uses function stored in the /models/process\_data file. Look under usage for training and testing these models.. 
- A settings file where various parameters can be configured. This is also where file locations for weights/biases, percentages, training and testing data, are loaded and/or saved. It can also be declared if the full or subset dataset is being used, if debugging needs to be turned on, or if certain weights need to be tested.
- A train\_models file where the all three of the models or specific models can be trained using the already generated training data. Further functionality needs to be added and the best place to train would be using the main.py file, although it train/tests all three.
- A test\_models file where each model's weight data is loaded and then and then used to test it. The sensitivity, specificity, and correctness of the model is then saved to a text file specified in the settings file.
- A collect\_weights file where the weight matrixes for each model are stored for later analysis. They are stored as text files under the /text\_files/[Experiment name]/ directory. 
- An analyze file where various analysis tools are stored. Here, the data stored in a text file from several runs is read, and then distributions of that data can be made to see how weight importance is distributed. Those distributions can then be plotted. The average sensitivity, specificity, and correctness can also be calculated here. 
- A gene\_analysis file for analyzing the individual genes. A lot of work remains in this file and it is mostly a mess. 
- Finally, a main file where all of the above functions are brought into one seamless process. After specifying settings, file locations, and save locations in the settings file, this file can be run with a command line argument of how many times it needs to train, test, and collect data. At the end, it will output the percentages of the runs.   
	- Contains another function that will create the text files needed to run the functions and other processes. More on that below in Usage. 
- A gene scraper file to find alternate names for genes in a given gene group database. It will make a normal google search for the gene and more often, the first result is the correct gene name. It saves this and uses the information in other files.
- A plot file to visualize the neural networks. This will allow us to see the nodes and connections of the network and how the graphs differ between models.
- Various sub-directories containing text files of unaltered data, training data, testing data, processed data, and the saved training weights of the models. 

## Prerequisites and installation
It is necessary to install various items before the files can be run. These include:
- For web scraping: bs4, requests
- For the neural networks: pickle, torch, pytorch, tqdm
- For plotting the networks: matplotlib, numpy, networkx
- For all files: datetime, time, random, collections, copy
- For testing: fraction

The required items to install are pickle, torch, pytorch, tqdm, matplotlib, numpy, datetime, random, time, collections, copy, fractions. Besides torch and pytorch, all of these can be install with `pip install [package]`

Install pytorch and torch with: 
```
pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Go to http://software.broadinstitute.org/gsea/downloads.jsp, login using an email and download the Hallmark gene sets, gene symbols file and save it as a text file under the /text\_files/ directory.

Download the normal/tumor dataset and store it as a text file under the /text\_files/ directory.

Download the gene\_pairs.pickle as their are various genes in the hallmark data base that use different names in the TCGA data. This file helps account for those differences. Save in the same place as the other two files. 

## Usage
All of these commands must be executed from the main directory.

First, set up all of the text files that are used by running:
```
python3 main.py create_files
```
Then make sure all of the desired settings and file locations are ok in settings.py. Otherwise, change then there. Some of the settings are redundant or need to be changed since they were part of older experiments and are probably no longer needed.

To collect data from train/testing the models, or specified models, multiple times, run: 
```
python3 main.py [iteration count] ["all or leave blank for all three] or any combination of ["split", "dense", "zerow"]
```
Or, to just train all three models, run:
```
python3 train_models.py ["all" or leave blank for all three] or any combination of ["split", "dense", "zerow"]
```
And, to test the models, run:
```
python3 test_models.py ["all" or leave blank for all three] or any combination of ["split", "dense", "zerow"]
```
Note individually training and testing will display a progress bar.
After collecting a large amount of data, to get gene group statistics run:
```
python3 analyze.py [Crit_A] [Crit_B] [Crit_C]
```
which will display a list of the hidden nodes and their calculated criteria values followed by a smaller list of the genes that satisfy the given criteria cutoff values and their average negative and positive weight values. 

## Background
Part of the [SRI International REU program](https://www.sri.com/careers/research-experience-undergraduates-program).  
