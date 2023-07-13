import multiprocessing

import numpy as np
import pandas as pd
import ray
import tensorflow as tf
from numpy import ndarray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import AUC, Precision, Recall

ray.init(log_to_driver=False)

np.random.seed(1)
tf.random.set_seed(2)

cores = multiprocessing.cpu_count()


class Dataset:
    def __init__(self, name: str, train_data: str, train_labels: str, dev_data: str, dev_labels: str):
        # Configurar un nombre para el dataset para que aparezca en los LOGS
        self.name = name

        # Cargamos las instancias vectorizadas y obtenemos las labels
        self.x_train: ndarray = np.load(train_data)
        self.y_train = tf.one_hot(pd.read_csv(train_labels)['Label'], 8)
        self.x_dev: ndarray = np.load(dev_data)
        self.y_dev = tf.one_hot(pd.read_csv(dev_labels)['Label'], 8)

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
    # Get dataset (features and classes)
    x_train, y_train = config['dataset'].train_data

    # ---------------------------------------------------------------------
    # Define de model
    model = Sequential()

    # Input layer and first hidden layers
    model.add(Dense(units=config['n_units'], activation=config['act_funct'], input_dim=x_train.shape[1]))
    model.add(Dropout(config['dropout']))

    # Add more hidden layers
    for _ in range(1, config['n_hidden_layers']):
        model.add(Dense(units=config['n_units'], activation=config['act_funct']))
        model.add(Dropout(config['dropout']))

    # Output layer
    model.add(Dense(units=8, activation='softmax'))

    # ---------------------------------------------------------------------

    # Configure early stopping using validation weighted f1 score as reference
    early_stop = EarlyStopping(monitor='val_weighted_f1', min_delta=0.001, patience=10, mode='max', restore_best_weights=True)

    # Compile the model and set the desired metrics
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
    model.fit(x_train, y_train, epochs=500, batch_size=config['batch_size'], callbacks=[early_stop], validation_data=config['dataset'].validation_data, verbose=0)

    # Use the best configuration to predict dev dataset and report to ray-tune (we should (must) add the optimum epoch number too if we use early stoping, but I forgot that...)
    tune.report(**model.evaluate(*config['dataset'].validation_data, return_dict=True))


def main():
    # Load the datasets
    lda_dataset = Dataset('lda_dataset', '../../data/vectorized/lda/lda_dataset.train.npy', '../../data/dataset/dataset.train.csv', '../../data/vectorized/lda/lda_dataset.dev.npy', '../../data/dataset/dataset.dev.csv')
    doc2vec_little = Dataset('Doc2Vec_Little', '../../data/vectorized/doc2vec_little/doc2vec_little.train.npy', '../../data/dataset/dataset.train.csv', '../../data/vectorized/doc2vec_little/doc2vec_little.dev.npy', '../../data/dataset/dataset.dev.csv')
    doc2vec_big = Dataset('Doc2Vec_Big', '../../data/vectorized/doc2vec_big/doc2vec_big.train.npy', '../../data/dataset/dataset.train.csv', '../../data/vectorized/doc2vec_big/doc2vec_big.dev.npy', '../../data/dataset/dataset.dev.csv')

    # ------------------
    # Set up ray tune

    # Algorithm to search the best hiper-parameter configuration (better than grid search)
    hyperopt_search = HyperOptSearch(metric="weighted_f1", mode="max")

    # Set up the experiments and run them with ray-tune
    analysis = tune.run(
        training_function,
        num_samples=500,
        max_failures=3,
        search_alg=hyperopt_search,
        local_dir="../../ray-logging/mlp_model_trainer",  # Importante para tener controlado la carpeta del logging
        config={
            'dataset': tune.choice([lda_dataset, doc2vec_little, doc2vec_big]),

            'optimizer': tune.choice(['Adam', 'Adamax']),

            'act_funct': tune.choice(['relu', 'tanh']),
            'n_units': tune.choice([i for i in range(4, 133, 8)]),
            'n_hidden_layers': tune.choice([1, 2, 3]),
            'dropout': tune.quniform(0.2, 0.5, 0.1),

            'batch_size': tune.choice([8, 16, 32])
        })

    print("Best config: ", analysis.get_best_config(metric="weighted_f1", mode="max"))


if __name__ == '__main__':
    main()
