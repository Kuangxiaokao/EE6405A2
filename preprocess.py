'''
Author: BugCrown
Date: 2024-10-25 19:59:49
LastEditors: BugCrown
'''
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import re
import spacy
import os
from unidecode import unidecode


# All datasets are downloaded from HuggingFace

# Load dataset
def LoadDataset(file_name):
    parquet_file = pq.ParquetFile(file_name)
    data = parquet_file.read().to_pandas()
    # Show the structure
    print(data.head())
    return data

# Regularization
def Regularization(text):
    text = unidecode(text)
    text = re.sub('(<.*?>)', ' ', text)
    text = re.sub('[,\.!?:()"]', ' ', text)
    text = re.sub('[^a-zA-Z"]',' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text


# Lemmatization
def Lemmatization(text):
    lemmatized_texts = []
    for doc in nlp.pipe(text, batch_size=64, n_process=12):
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        lemmatized_texts.append(lemmatized_text)
    return lemmatized_texts

# Write processed dataset to a new file
def WriteDataset(data, file_name):
    data.to_parquet(file_name, engine='pyarrow')
    
# Preprocess 
def Preprocess(dataset, input_file, output_file):
    dir = dataset + '_process'
    os.makedirs(dir, exist_ok=True)
    if dataset == 'sst2':
        data = LoadDataset(input_file)
        data['sentence'] = data['sentence'].apply(Regularization)
        print(data.head())
        data['sentence'] = Lemmatization(data['sentence'].tolist())
        WriteDataset(data, output_file) 
    elif dataset == 'imdb' or 'yelp':
        data = LoadDataset(input_file)
        if dataset == 'yelp':
            sample_size = len(data)
            # data = data.sample(n=sample_size, random_state=42)
        data['text'] = data['text'].apply(Regularization)
        print(data.head())
        data['text'] = Lemmatization(data['text'].tolist())
        WriteDataset(data, output_file)        
    print(data.head()) 

if __name__ == '__main__':    
    nlp = spacy.load("en_core_web_sm")
    # # SST2: https://huggingface.co/datasets/stanfordnlp/sst2
    # Preprocess('sst2', './sst2/data/train-00000-of-00001.parquet', './sst2_process/train.parquet')
    # Preprocess('sst2', './sst2/data/validation-00000-of-00001.parquet', './sst2_process/validation.parquet')
    # Preprocess('sst2', './sst2/data/test-00000-of-00001.parquet', './sst2_process/test.parquet')
    # # IMDB: https://huggingface.co/datasets/stanfordnlp/imdb
    # Preprocess('imdb', './imdb/plain_text/train-00000-of-00001.parquet', './imdb_process/train.parquet')
    # Preprocess('imdb', './imdb/plain_text/test-00000-of-00001.parquet', './imdb_process/test.parquet')
    # Preprocess('imdb', './imdb/plain_text/unsupervised-00000-of-00001.parquet', './imdb_process/unsupervised.parquet')
    # YELP: https://huggingface.co/datasets/contemmcm/yelp_review
    Preprocess('yelp', './yelp_review_full/yelp_review_full/train-00000-of-00001.parquet', './yelp_process/train.parquet')
    Preprocess('yelp', './yelp_review_full/yelp_review_full/test-00000-of-00001.parquet', './yelp_process/test.parquet')
