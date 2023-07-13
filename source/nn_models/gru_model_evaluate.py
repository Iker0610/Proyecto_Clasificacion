import multiprocessing

import numpy as np
import pandas as pd
import ray
import tensorflow as tf
from gensim.models import KeyedVectors
from numpy import ndarray
from ray import tune
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Embedding, Dense, Bidirectional, GRU
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow_addons.metrics import F1Score

ray.init(log_to_driver=False)

np.random.seed(1)
tf.random.set_seed(2)

cores = multiprocessing.cpu_count()


class Dataset:
    def __init__(self, name: str, train_data: str, dev_data: str, test_data: str, embeddings_path: str):
        # Configurar un nombre para el dataset para que aparezca en los LOGS
        self.name = name

        # Cargamos los datos
        train = pd.read_csv(train_data)
        dev = pd.read_csv(dev_data)
        test = pd.read_csv(test_data)
        embeddings_keyed: KeyedVectors = KeyedVectors.load(embeddings_path)

        # Los procesamos y hacemos padding a los textos para que tengan la misma longitud por batch
        self.embeddings: ndarray = np.append(np.zeros((1, embeddings_keyed.vector_size)), embeddings_keyed.get_normed_vectors(), axis=0)

        self.x_train: ndarray = preprocessing.sequence.pad_sequences([[embeddings_keyed.get_index(token) + 1 for token in text.split() if embeddings_keyed.has_index_for(token)] for text in train['Tokenized_Text']], truncating='post', maxlen=175)
        self.y_train: ndarray = np.asarray(tf.one_hot(train['Label'], 8))

        self.x_dev: ndarray = preprocessing.sequence.pad_sequences([[embeddings_keyed.get_index(token) + 1 for token in text.split() if embeddings_keyed.has_index_for(token)] for text in dev['Tokenized_Text']], truncating='post', maxlen=175)
        self.y_dev: ndarray = np.asarray(tf.one_hot(dev['Label'], 8))

        self.x_train_dev: ndarray = np.concatenate((self.x_train, self.x_dev), axis=0)
        self.y_train_dev: ndarray = np.concatenate((self.y_train, self.y_dev), axis=0)

        self.x_test: ndarray = preprocessing.sequence.pad_sequences([[embeddings_keyed.get_index(token) + 1 for token in text.split() if embeddings_keyed.has_index_for(token)] for text in test['Tokenized_Text']], truncating='post', maxlen=175)
        self.y_test: ndarray = np.asarray(tf.one_hot(test['Label'], 8))

        # Stratified
        self.skf = StratifiedKFold(n_splits=10, shuffle=True)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def stratified_kfold(self):
        return self.skf.split(self.x_train_dev, tf.argmax(self.y_train_dev, axis=1))

    def train_data(self, indexes: list):
        return self.x_train_dev[indexes], self.y_train_dev[indexes]

    def validation_data(self, indexes: list):
        return self.x_train_dev[indexes], self.y_train_dev[indexes]

    @property
    def test_data(self):
        return self.x_test, self.y_test

    def val_test_data(self, indexes: list):
        return np.concatenate((self.x_test, self.x_train_dev[indexes]), axis=0), np.concatenate((self.y_test, self.y_train_dev[indexes]), axis=0)


def training_function(config):
    # Get train and validation splits
    train_split, val_split = config['splits']

    # Get training data
    x_train, y_train = config['dataset'].train_data(train_split)

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
    model.fit(x_train, y_train, epochs=600, batch_size=config['batch_size'], callbacks=[early_stop], validation_data=config['dataset'].validation_data(val_split), verbose=1)

    # Evaluate over train
    train_results = {'evaluated_split': 'train'}
    train_results.update(model.evaluate(*config['dataset'].train_data(train_split), return_dict=True))
    tune.report(**train_results)

    # Evaluate over val + test
    val_test_results = {'evaluated_split': 'val_test'}
    val_test_results.update(model.evaluate(*config['dataset'].val_test_data(val_split), return_dict=True))
    tune.report(**val_test_results)

    # Evaluate over test
    test_results = {'evaluated_split': 'test'}
    test_results.update(model.evaluate(*config['dataset'].test_data, return_dict=True))
    tune.report(**test_results)


def main():
    # Load datasets
    doc2vec_little = Dataset('Doc2Vec_Little', '../../data/dataset/dataset.train.csv', '../../data/dataset/dataset.dev.csv', '../../data/dataset/dataset.test.csv', '../../data/embeddings/doc2vec_little.embeddings')
    doc2vec_big = Dataset('Doc2Vec_Big', '../../data/dataset/dataset.train.csv', '../../data/dataset/dataset.dev.csv', '../../data/dataset/dataset.test.csv', '../../data/embeddings/doc2vec_big.embeddings')

    # ------------------
    # Set up ray tune

    # Run the best model for every k-fold cross validation split
    tune.run(
        training_function,
        num_samples=1,
        max_failures=3,
        local_dir="../../ray-logging/gru_model_evaluation",
        config={
            'dataset': doc2vec_big,

            'splits': tune.grid_search(list(doc2vec_big.stratified_kfold)),

            'optimizer': 'Adamax',

            'stacked_gru_layers': True,
            'n_units_gru': 64,
            'bidirectional': True,
            'second_layer_bidirectional': True,
            'trainable_embeddings': False,

            'act_funct': 'relu',
            'n_units_mlp': 64,
            'n_hidden_layers': 3,

            'dropout': 0.25,
            'batch_size': 32
        })


if __name__ == '__main__':
    main()
