# [Embeddings](https://docs.google.com/presentation/d/1xOXrY8AvHEO6jKHqcslv0AM-RYvxf401M8GA9K0tI9M/edit?usp=sharing)

This work has been done as part of a project at **Insight Data Science** as an *AI fellow*.

## Introduction
The project was targeted at representing and understanding abstract data like words in text, items in inventory, transactions etc. which traditioanlly are represented using unique IDs but that method fails to capture the meaning or properties about the data. Capturing meaning for a word, property about an item in inventory or transaction can be useful to find similar words in text, similar items or similar transactions in the data.

Embedding is a vector of real numbers which is used to represent an item in D dimensions and has become the most useful way of representation in Machine Learning world. These embeddings can also store the meaning and properties of data. Being vectors of real numbers, performing algebraic operations like calculating distance helps us to understand how close or similar two items are in their meaning/features or calculating direction between two group of items gives a sense of which items pair best between the two groups. 

In this project I built a generic pipeline for sequential data having context, such that leveraging context I capture the features of items in the data. I experimented my pipeline on text and E-commerce dataset 

Additional information is available from my presentation present as hyperlink.
  
## Pipeline
I have divided my pipeline into following steps:
### Pre-processing
For the pre-processing step I have written an abstract python class called **Preprocess** which can be inherited to write a custom python script to pre-process raw data into sequential data. An example is already present `preprocess_raw.py` on how to implement custom class. By default the processed data is stored in data folder.
### Training
After pre-processing, the model can be trained to get embeddings for each item present in the data. The pipeline not only generates embeddings over the data but also plots the items using **TNSE** for visualizing the distribution of the data on the basis of feature similarity. 
### Inferring
There is small flask application for doing the following:
1. Get all items that have been processed from the raw data
2. Search the presence of a particular item in the data
3. Get embeddings for all data items to use somewhere else
4. Find top k similar items given input an item and number k

## Model
I am using **word2vec skip-gram** model developed by *Mikolov et. al.* for capturing meaning through embeddings. The paper describing the model can be found [here](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

## Dependencies  
The project depends on the following main packages:
1. python 2.7
2. tensorflow
3. numpy
4. scikit-learn
5. matplotlib
6. flask

TODO: all dependencies packages have to be aggregated as requirements.py for easy set-up

## Run
After cloning the repository all commands and scripts are run from directory **Embeddings** 
### Pre-process
After customizing **preprocess_raw.py** script, the following command can be run to start pre-processing

`$ python preprocess_raw.py`
### Training
For training or pre-processing + training, the follwing script can be run

`$ sh run.sh`

The script can be modified accordingly for customing arguments to model
### Inferring
To use the embeddings and find similarities over items, the flask application has to be started as follows

`$ sh findsimilar.sh`
