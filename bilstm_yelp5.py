'''
Author: BugCrown
Date: 2024-10-28 13:07:43
LastEditors: BugCrown
'''
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical

# Load dataset
def LoadDataset(file_name):
    parquet_file = pq.ParquetFile(file_name)
    data = parquet_file.read().to_pandas()
    # Show the structure
    print(data.head())
    return data

train_data = LoadDataset('./data/yelp_process/train.parquet')
test_data = LoadDataset('./data/yelp_process/test.parquet')

train_texts = train_data['text']
train_labels = to_categorical(train_data['label'], num_classes=5)

validation_texts = test_data['text']
validation_labels = to_categorical(test_data['label'], num_classes=5)

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=1000,
    pad_to_max_tokens=True
)
with tf.device('/CPU:0'):
    dataset = tf.data.Dataset.from_tensor_slices(train_texts).batch(16)
    encoder.adapt(dataset)
    
    encoded_batches = []
    for batch in dataset:
        encoded_batch = encoder(batch)
        encoded_batches.append(encoded_batch)
    encoded_train_texts = tf.concat(encoded_batches, axis=0)
    encoded_validation_texts = encoder(validation_texts)

# Create the model
model = tf.keras.Sequential([
    Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True),
    # Bidirectional(LSTM(128, return_sequences=True)),
    # Dropout(0.5),
    # Bidirectional(LSTM(128, return_sequences=True)),
    # Dropout(0.5),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])

# To implement early stopping
earlystopping = callbacks.EarlyStopping(monitor ="accuracy", 
                                        mode ="max", patience = 5, 
                                        restore_best_weights = True)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.5),
              metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

historyLSTM = model.fit(encoded_train_texts, train_labels, epochs=10,
                    validation_data=(encoded_validation_texts, validation_labels),
                    batch_size=64,
                    callbacks=[earlystopping])
with open('training_history_yelp5.json', 'w') as f:
    json.dump(historyLSTM.history, f)
model.save('bilstm_yelp5.h5')

def plot_graphs(name, history, metric):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.savefig(name)

plot_graphs("yelp.png", historyLSTM, "accuracy")