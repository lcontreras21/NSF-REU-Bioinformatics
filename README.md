# Neural Network REU Project

## Contents
..* Three models: a fully connected (dense) model; a partially connected (zero-weights) model; a sparsely conncetd (split) model
..* A gene scraper file to find alternate names for genes in gene group data. It will make a normal google search for the gene and more often, the first result is the correct gene name. It saves this and uses the information in other files.
..* A plot file to visualize the neural networks. This will allow us to see the nodes and connections of the network and how the graph differs between models.
..* A settings and test models file where various parameters can be configured and then the models can be trained and tested either individually or collectively.
..* Various sub-directories containing text files of unaltered data, training data, testing data, processed data, and the saved training weights of the models. 

## Installation
It is necessary to install various items before the files can be run. These include:
..* For web scraping: bs4, requests
..* For the neural networks: pickle, torch, pytorch, tqdm
..* For plotting the networks: matplotlib, networkx, numpy

## Background
Part of the [SRI International REU program](https://www.sri.com/careers/research-experience-undergraduates-program). The student, Luis Contreras-Orendain, is a rising junior at Haverford College and will be working with Subarna Sinha. 

#### Development of Machine Learning Models for Cancer Driver Identification
Large-scale cancer genome sequencing projects have resulted in the discovery of numerous cancer-specific genetic changes. However, it remains unclear how many of these genetic changes contribute to tumorigenesis.

In this project, the goal will be to leverage large omics databases and existing biological knowledge to learn an interpretable machine learning model that can differentiate between cancer samples and normal samples based on these genetic changes. Unlike previous black box models, the goal is to build a model that will be interpretable and will enable mechanistic interpretation by explaining how specific genetic changes contribute to individual cancer processes. The student’s project will contribute to a larger effort for cancer drug target discovery.

Outcome: The student will learn how to build a machine learning model to solve a key problem in cancer research. Many of the techniques and tools learned during this project can be applied to other machine learning projects. Ultimately, the student’s work will lead to the development of an important tool for the research community.
