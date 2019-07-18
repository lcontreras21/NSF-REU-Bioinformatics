# Bioinformatics NSF REU Project

## Title and Description
###### Development of Machine Learning Models for Cancer Driver Identification
Large-scale cancer genome sequencing projects have resulted in the discovery of numerous cancer-specific genetic changes. However, it remains unclear how many of these genetic changes contribute to tumorigenesis.

In this project, the goal will be to leverage large omics databases and existing biological knowledge to learn an interpretable machine learning model that can differentiate between cancer samples and normal samples based on these genetic changes. Unlike previous black box models, the goal is to build a model that will be interpretable and will enable mechanistic interpretation by explaining how specific genetic changes contribute to individual cancer processes. The student’s project will contribute to a larger effort for cancer drug target discovery.

Outcome: The student will learn how to build a machine learning model to solve a key problem in cancer research. Many of the techniques and tools learned during this project can be applied to other machine learning projects. Ultimately, the student’s work will lead to the development of an important tool for the research community.

## Installation
It is necessary to install various items before the files can be run. These include:
- For web scraping: bs4, requests
- For the neural networks: pickle, torch, pytorch, tqdm
- For plotting the networks: matplotlib, networkx, numpy
- For all files: datetime, time, random, collections, copy

## Contents
- Three models: a fully connected (dense) model; a partially connected (zero-weights) model; a sparsely connected (split) model. Each model uses function stored in the process\_data file. To train individual models, they must be ran from the main.py file, otherwise erros from import issues will come up. 
- A settings file where various parameters can be configured. This is also where file locations for weights/biases, percentages, training and testing data, are loaded and/or saved. It can also be declared if the full or subset dataset is being used, if debugging needs to be turned on, or if certain weights need to be tested.
- A test\_models file where each model's weight data is loaded and then and then used to test it. The sensitivity, specificity, and correctness of the model is then saved to a text file specified in the settings file.
- A collect\_weights file that is used after the training process is done to collect the top n weights and biases for each model. Currently, only collecting the top 5 biggest weights. The data is stored in a text file and later used for analysis and statistics. 
- An analyze file where various analysis tools are stored. Here, the data stored in a text file from several runs is read, and then distributions of that data can be made to see how weight importance is distributed. Those distributions can then be plotted. If weights have been removed for testing, the difference in performance can be calculated and shown here. The average sensitivity, specificity, and correctness can also be calculated here. 
- Finally, a main file where everything is brought into one seamless process. After specifiying settings, file locations, and save locations in the settings file, this file can be run with a command line argument of how many times it needs to train, test, and collect data. At the end, it will output the percentages of the runs, draw the distribution data, and, if weights were being tested, check that none of the data has those weights. 
	- After two sessions have been run, one where all weights were present and another where some weights were removed, the difference can be calculated and shown in the analyze file as mentioned above. 
- A gene scraper file to find alternate names for genes in a given gene group database. It will make a normal google search for the gene and more often, the first result is the correct gene name. It saves this and uses the information in other files.
- A plot file to visualize the neural networks. This will allow us to see the nodes and connections of the network and how the graphs differs between models.
- In the preprocessing folder, there is a file to make subset datasets based on the gene groups, make training and testing data from the full dataset and from the subset dataset just created. It can also make other necessary files such as files containing the names of the genes in the full and subset datasets necessary for processing the data when training. 
- Various sub-directories containing text files of unaltered data, training data, testing data, processed data, and the saved training weights of the models. 

## Background
Part of the [SRI International REU program](https://www.sri.com/careers/research-experience-undergraduates-program).  
