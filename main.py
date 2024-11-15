from preprocess import LoadDataset
import numpy as np
from keras.src.utils.module_utils import tensorflow
import tensorflow as tf
def to_categorical(labels, num_classes=None):
    labels = np.array(labels, dtype='int')
    if num_classes is None:
        num_classes = np.max(labels) + 1
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def preparing_encoder(training_set, VOCAB_SIZE=1000):
    """
    take in training dataset and vectorize it into int
    high frequency tocken will get an int close to 1
    VOCAB_SIZE: the most frequntly appealing token will be vectorized
    if a token is less frequent and falls outside of the top 1000 words ,
    it will typically be assigned a special out-of-vocabulary (OOV)
    return an adapted encoder for vectorization
    """
    # reviews = training_set['review'].tolist()
    reviews = training_set['text'].tolist()
    encoder = tensorflow.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        pad_to_max_tokens=True  # This argument no longer exists in TensorFlow 2.17
    )
    print('start adapting')
    encoder.adapt(reviews)
    print('finish adapting')

    vocab = np.array(encoder.get_vocabulary())

    # print(vocab[:20])
    # print(training_set.text.iloc[0])
    #
    # encoded_example = encoder(training_set.text.iloc[3]).numpy()
    # print(encoded_example)
    return encoder


if __name__ == '__main__':

    net = 'rnn'
    # datset = 'imdb'
    # data_train = LoadDataset(file_name='./imdb_process/train.parquet')
    # data_test = LoadDataset(file_name='./imdb_process/test.parquet')
    # data_train_label = data_train['label']
    # data_test_label = data_test['label']
    #
    # datset = 'sst2'
    # data_train = LoadDataset(file_name='./sst2_process/train.parquet')
    # data_test = LoadDataset(file_name='./sst2_process/validation.parquet')
    #
    datset = 'yelp'
    data_train = LoadDataset(file_name='./yelp_process/train.parquet')
    data_train_label = to_categorical(data_train['label'], num_classes=5)
    data_test = LoadDataset(file_name='./yelp_process/test.parquet')
    data_test_label = to_categorical(data_test['label'], num_classes=5)
    print(type(data_train))
    print(data_train.columns)
    print(type(data_test))
    print(data_test.columns)

    gpus = tensorflow.config.list_physical_devices('GPU')
    print('number of GPU available: ', len(gpus))


    encoder = preparing_encoder(training_set=data_train)
    if net == 'lstm':
        model = tensorflow.keras.Sequential([
            encoder,
            tensorflow.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),  # Use encoder vocabulary size as input dimension
                output_dim=64,
                # Use masking to handle variable-length sequences
                mask_zero=True
            ),
            tensorflow.keras.layers.LSTM(64),  # LSTM layer with 64 units
            tensorflow.keras.layers.Dense(5) if datset=='yelp' else tensorflow.keras.layers.Dense(1) # Output layer for regression or binary classification
        ])
    elif net == 'gru':
        model = tensorflow.keras.Sequential([
            encoder,
            tensorflow.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),  # Use encoder vocabulary size as input dimension
                output_dim=64,
                # Use masking to handle variable-length sequences
                mask_zero=True
            ),
            tensorflow.keras.layers.GRU(64),  # LSTM layer with 64 units
            tensorflow.keras.layers.Dense(5) if datset=='yelp' else tensorflow.keras.layers.Dense(1)  # Output layer for regression or binary classification
        ])
    elif net == 'rnn':
        model = tensorflow.keras.Sequential([
            encoder,
            tensorflow.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),  # Use encoder vocabulary size as input dimension
                output_dim=64,
                # Use masking to handle variable-length sequences
                mask_zero=True
            ),
            tensorflow.keras.layers.SimpleRNN(64),  # LSTM layer with 64 units
            tensorflow.keras.layers.Dense(5) if datset=='yelp' else tensorflow.keras.layers.Dense(1)  # Output layer for regression or binary classification
        ])
    else:
        raise ValueError('Invalid network type')

    earlystopping = tensorflow.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                             mode="max", patience=5,
                                                             restore_best_weights=True)

    # Initialize CSVLogger to record training progress
    import  time
    training_start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    csv_logger = tensorflow.keras.callbacks.CSVLogger(net+'_'+datset+f'_training_log{training_start_time}.csv', append=True)

    model.compile(
        loss = tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True) if datset == 'yelp' else tensorflow.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tensorflow.keras.optimizers.Adam(clipvalue=0.5),
        metrics=['accuracy', 'Precision', 'Recall', 'AUC']
    )

    trained_model = model.fit(
        data_train['sentence'] if datset == 'sst2' else data_train['text'], data_train_label,
        epochs=10,
        validation_data=(data_test['sentence'] if datset == 'sst2' else data_train['text'], data_test_label),
        batch_size=32,
        callbacks=[earlystopping, csv_logger]
    )

