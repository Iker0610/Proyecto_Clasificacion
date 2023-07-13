import multiprocessing
from pprint import pprint

import numpy as np
import pandas as pd
import ray
import tensorflow as tf
from gensim.models import KeyedVectors
from numpy import ndarray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Embedding, Dense, Bidirectional, GRU
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow_addons.metrics import F1Score

ray.init(log_to_driver=False)

np.random.seed(1)
tf.random.set_seed(2)

cores = multiprocessing.cpu_count()


class Dataset:
    def __init__(self, name: str, train_data: str, dev_data: str, embeddings_path: str):
        # Configurar un nombre para el dataset para que aparezca en los LOGS
        self.name = name

        # Cargamos los datos
        train = pd.read_csv(train_data)
        dev = pd.read_csv(dev_data)
        embeddings_keyed: KeyedVectors = KeyedVectors.load(embeddings_path)

        # Los procesamos y hacemos padding a los textos para que tengan la misma longitud por batch
        self.embeddings: ndarray = np.append(np.zeros((1, embeddings_keyed.vector_size)), embeddings_keyed.get_normed_vectors(), axis=0)
        self.x_train: ndarray = preprocessing.sequence.pad_sequences([[embeddings_keyed.get_index(token) + 1 for token in text.split() if embeddings_keyed.has_index_for(token)] for text in train['Tokenized_Text']], truncating='post', maxlen=175)
        self.y_train = tf.one_hot(train['Label'], 8)
        self.x_dev: ndarray = preprocessing.sequence.pad_sequences([[embeddings_keyed.get_index(token) + 1 for token in text.split() if embeddings_keyed.has_index_for(token)] for text in dev['Tokenized_Text']], truncating='post', maxlen=175)
        self.y_dev = tf.one_hot(dev['Label'], 8)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def train_data(self):
        return self.x_train, self.y_train

    @property
    def validation_data(self):
        return self.x_dev, self.y_dev


def training_function(config):
    # Get training data
    x_train, y_train = config['dataset'].train_data

    # Get embeddings
    embeddings = config['dataset'].embeddings

    # Define de model
    model = Sequential()

    # Add embedding layer
    model.add(Embedding(*embeddings.shape, mask_zero=True))

    # Add LSTM layer
    if config['stacked_gru_layers']:
        if config['bidirectional']:
            model.add(Bidirectional(GRU(config['n_units_gru'], return_sequences=True)))
        else:
            model.add(GRU(config['n_units_gru'], return_sequences=True))
        model.add(Dropout(config['dropout']))

        if config['second_layer_bidirectional']:
            model.add(Bidirectional(GRU(config['n_units_gru'])))
        else:
            model.add(GRU(config['n_units_gru']))
        model.add(Dropout(config['dropout']))

    else:
        if config['bidirectional']:
            model.add(Bidirectional(GRU(config['n_units_gru'])))
        else:
            model.add(GRU(config['n_units_gru']))
        model.add(Dropout(config['dropout']))

    # Add hidden layers over LSTM
    for _ in range(config['n_hidden_layers']):
        model.add(Dense(units=config['n_units_mlp'], activation=config['act_funct']))
        model.add(Dropout(config['dropout']))

    # Output layer
    model.add(Dense(units=8, activation='softmax'))

    # ---------------------------------------------------------------------

    # Add pretrained embeddings:
    model.layers[0].set_weights([embeddings])
    model.layers[0].trainable = config['trainable_embeddings']

    # Configure early stopping using validation f1 score as reference
    early_stop = EarlyStopping(monitor='val_weighted_f1', min_delta=0.0005, patience=10, mode='max', restore_best_weights=True)

    # Compile the model using a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer=config['optimizer'], metrics=['categorical_accuracy',

                                                                                           F1Score(num_classes=8, average='weighted', name='weighted_f1'),
                                                                                           F1Score(num_classes=8, average='micro', name='micro_f1'),
                                                                                           F1Score(num_classes=8, average='macro', name='macro_f1'),

                                                                                           Precision(class_id=0, name='Precision_0'), Recall(class_id=0, name='Recall_0'),
                                                                                           Precision(class_id=1, name='Precision_1'), Recall(class_id=1, name='Recall_1'),
                                                                                           Precision(class_id=2, name='Precision_2'), Recall(class_id=2, name='Recall_2'),
                                                                                           Precision(class_id=3, name='Precision_3'), Recall(class_id=3, name='Recall_3'),
                                                                                           Precision(class_id=4, name='Precision_4'), Recall(class_id=4, name='Recall_4'),
                                                                                           Precision(class_id=5, name='Precision_5'), Recall(class_id=5, name='Recall_5'),
                                                                                           Precision(class_id=6, name='Precision_6'), Recall(class_id=6, name='Recall_6'),
                                                                                           Precision(class_id=7, name='Precision_7'), Recall(class_id=7, name='Recall_7')])

    # Train the model
    model.fit(x_train, y_train, epochs=600, batch_size=config['batch_size'], callbacks=[early_stop], validation_data=config['dataset'].validation_data, verbose=1)

    # Use the best configuration to predict dev dataset and report to ray-tune
    tune.report(**model.evaluate(*config['dataset'].validation_data, return_dict=True))


def main():
    # Load datasets
    doc2vec_little = Dataset('Doc2Vec_Little', '../../data/dataset/dataset.train.csv', '../../data/dataset/dataset.dev.csv', '../../data/embeddings/doc2vec_little.embeddings')
    doc2vec_big = Dataset('Doc2Vec_Big', '../../data/dataset/dataset.train.csv', '../../data/dataset/dataset.dev.csv', '../../data/embeddings/doc2vec_big.embeddings')

    # ------------------
    # Set up ray tune

    # Algorithm to search the best hiper-parameter configuration (better than grid search)
    hyperopt_search = HyperOptSearch(metric="weighted_f1", mode="max")

    # Set up the experiments and run them with ray-tune
    analysis = tune.run(
        training_function,
        num_samples=200,
        max_failures=3,
        search_alg=hyperopt_search,
        local_dir="../../ray-logging/gru_model_trainer",
        config={
            'dataset': tune.choice([doc2vec_big, doc2vec_little]),

            'optimizer': 'Adamax',

            'stacked_gru_layers': tune.choice([True, False]),
            'n_units_gru': tune.choice([2 ** i for i in range(3, 7)]),
            'bidirectional': tune.choice([True, False]),
            'second_layer_bidirectional': tune.choice([True, False]),
            'trainable_embeddings': tune.choice([True, False]),

            'act_funct': tune.choice(['relu']),
            'n_units_mlp': tune.choice([2 ** i for i in range(3, 8)]),
            'n_hidden_layers': tune.choice([0, 1, 2, 3]),

            'dropout': tune.quniform(0.2, 0.5, 0.05),
            'batch_size': 32
        })

    print("Best config: ", analysis.get_best_config(metric="weighted_f1", mode="max"))


if __name__ == '__main__':
    main()
