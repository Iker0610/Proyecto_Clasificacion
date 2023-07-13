import argparse
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pandas import DataFrame

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

cores = multiprocessing.cpu_count()


def train_doc2vec(input_dataframe: Union[str, DataFrame], output_file: str = None, text_column: str = 'Text', features: int = 200, window: int = 5, epochs: int = 20, dm: int = 1, dbow_words: int = 0, min_count: int = 20):
    # Comprobamos si es un dataframe o un path, si es un path cargamos el archivo
    if isinstance(input_dataframe, str):
        input_dataframe = pd.read_csv(input_dataframe)

    # Preparamos los archivos en el formato requerido por gensim
    documents = [TaggedDocument(doc.split(), [i]) for i, doc in input_dataframe[text_column].iteritems()]

    # Preparamos el modelo y lo entrenamos
    model = Doc2Vec(vector_size=features, epochs=epochs, min_count=min_count, window=window, dm=dm, dbow_words=dbow_words, workers=cores)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    # Generamos la carpeta padre y guardamos el modelo
    if output_file:
        Path(output_file).parent.mkdir(exist_ok=True)
        model.save(output_file)

    return model


def vectorize_doc2vec(input_dataframe: Union[str, DataFrame], model: Union[str, Doc2Vec], text_column: str = 'Text', output_file: str = None):
    # Comprobamos si es un dataframe o un path, si es un path cargamos el archivo
    if isinstance(input_dataframe, str):
        input_dataframe = pd.read_csv(input_dataframe)

    # Comprobamos si es un modelo o un path, si es un path cargamos el modelo
    if isinstance(model, str):
        model = Doc2Vec.load(model)

    # Vectorizamos los textos
    vectorized_documents = np.asarray([model.infer_vector(doc.split()) for _, doc in input_dataframe[text_column].iteritems()])

    # Generamos la carpeta padre y guardamos el resultado
    if output_file:
        Path(output_file).parent.mkdir(exist_ok=True)
        np.save(output_file, vectorized_documents)

    return vectorized_documents


cmd_functions = {'train_doc2vec': train_doc2vec,
                 'vectorize_doc2vec': vectorize_doc2vec}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False, fromfile_prefix_chars='@')

    subparsers = parser.add_subparsers(dest='command', required=True, help='Choose a function to use.')

    # Train Tf-Idf arguments
    fit_doc2vec_args = subparsers.add_parser('train_doc2vec', help='Train a new doc2vec with the given data.')
    fit_doc2vec_args.add_argument('-i', '--input_dataframe', type=str, required=True, help='Input csv file containing training texts.')
    fit_doc2vec_args.add_argument('-o', '--output_file', type=str, required=True, help='Output file where model will be saved. IT MUST BE A FULL PATH, NOT RELATIVE.')
    fit_doc2vec_args.add_argument('-c', '--text_column', type=str, default='Text', required=False, help="Name of the data column containing the texts.")

    fit_doc2vec_args.add_argument('--features', type=int, default=200, required=False, help='Number of features/dimensions of the result vector.')
    fit_doc2vec_args.add_argument('--window', type=int, default=5, required=False, help='The maximum distance between the current and predicted word within a sentence.')
    fit_doc2vec_args.add_argument('--epochs', type=int, default=20, required=False, help='Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec.')
    fit_doc2vec_args.add_argument('--dm', type=int, default=1, required=False, help='Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.')
    fit_doc2vec_args.add_argument('--dbow_words', type=int, default=0, required=False, help='If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).')
    fit_doc2vec_args.add_argument('--min_count', type=int, default=20, required=False, help='Ignores all words with total frequency lower than this.')

    vectorize_doc2vec_args = subparsers.add_parser('vectorize_doc2vec', help='Train a new doc2vec with the given data.')
    vectorize_doc2vec_args.add_argument('-i', '--input_dataframe', type=str, required=True, help='Input CSV file containing training texts.')
    vectorize_doc2vec_args.add_argument('-o', '--output_file', type=str, required=True, help='Output file where tokenized text arrays will be saved.')
    vectorize_doc2vec_args.add_argument('-m', '--model', type=str, required=True, help="Name of the data column containing the texts.")
    vectorize_doc2vec_args.add_argument('-c', '--text_column', type=str, default='Text', required=False, help="Path to the pre-trained model file.")

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    # Parse args
    args = vars(parser.parse_args())

    # Run the main function
    cmd_functions[args.pop('command')](**args)
