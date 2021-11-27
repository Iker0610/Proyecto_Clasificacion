import argparse
import json
import os
from glob import glob as get_files
from typing import Union

import numpy as np
from joblib import dump as save_model, load as load_model
from numpy import ndarray
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm


def _generate_tfidf(data: [str], model: str, output: str, min_ngrams: int = 1, max_ngrams: int = 1,
                    min_df: Union[int, float] = 1, max_df: Union[int, float] = 0.9, max_features: int = None,
                    sublinear_tf: bool = False) -> ndarray:
    """
    Generates tf-idf and saves it. It also transforms data.

    Parameters
    ----------
    data : List of string
        List containing each corpus text file
        It's recommended they are pre tokenized
    model : str
        Folder where the generated models will be allocated / loaded
    output : str
        Folder where vectorized corpus will be allocated
    min_ngrams : int
        The lower boundary of the range of n-values for different n-grams to be extracted
    max_ngrams : int
        The upper boundary of the range of n-values for different n-grams to be extracted
    min_df : int or float
        Ignore terms that have a document count/frequency strictly lower than the given threshold when building Tf-Idf
    max_df : int or float
        Ignore terms that have a document count/frequency strictly higher than the given threshold when building Tf-Idf
    max_features : int
        If not None, Tf-Idf builds a vocabulary that only consider the top max_features ordered by term frequency across the corpus
    sublinear_tf : bool
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Returns
    -------
    vectorized_corpus: ndarray of shape (n_samples, n_features)
        Vectorized corpus
    """

    # Aplicamos el TF-IDF
    tf_idf_model = TfidfVectorizer(ngram_range=(min_ngrams, max_ngrams), max_df=max_df, min_df=min_df,
                                   sublinear_tf=sublinear_tf, max_features=max_features)
    vectorized_corpus_matrix = tf_idf_model.fit_transform(data).toarray()

    # Guardamos el modelo
    with open(model + os.path.sep + 'tf-idf_vectorizer.joblib', 'wb') as model_file:
        save_model(tf_idf_model, model_file)

    # Guardamos el texto vectorizado
    with open(output + os.path.sep + 'tfidf_vectorized_corpus.npy', 'wb') as file_handler:
        np.save(file_handler, vectorized_corpus_matrix)

    return vectorized_corpus_matrix


def _generate_lda(data: [str], model: str, output: str, iterations: int = 10000, topics: int = 200, alpha=None,
                  beta=None) -> ndarray:
    """
    Generates lda and saves it. It also transforms data.

    Parameters
    ----------
    data : List of str
        List containing each corpus text file
        It's recommended they are pre tokenized
    model : str
        Folder where the generated models will be allocated / loaded
    output : str
        Folder where vectorized corpus will be allocated
    iterations : int
        Max iterations for LDA
    topics : int
        Number of features/topics/dimensions of the result vector
    alpha : float
        Alpha value for LDA
    beta : float
        Beta value for LDA

    Returns
    -------
    vectorized_corpus: ndarray of shape (n_samples, n_features)
        Vectorized corpus
    """

    # Convertimos los textos a una matriz de frecuencia de tokens para el LDA
    tf_vectorizer = CountVectorizer()
    matriz_frequencias_corpus = tf_vectorizer.fit_transform(data)

    # Guardamos el modelo
    with open(model + os.path.sep + 'BoW_vectorizer.joblib', 'wb') as model_file:
        save_model(tf_vectorizer, model_file)

    # Aplicamos el LDA
    lda_model = LDA(n_components=topics, max_iter=iterations, learning_method='online', learning_offset=50.,
                    random_state=0,
                    doc_topic_prior=alpha, topic_word_prior=beta, n_jobs=-1, verbose=1).fit(matriz_frequencias_corpus)

    # Guardamos el modelo
    with open(model + os.path.sep + 'LDA_vectorizer.joblib', 'wb') as model_file:
        save_model(lda_model, model_file)

    # Predecimos los documentos para obtener los textos vectorizados
    vectorized_corpus_matrix = lda_model.transform(matriz_frequencias_corpus)

    # Guardamos el texto vectorizado
    with open(output + os.path.sep + 'lda_vectorized_corpus.npy', 'wb') as file_handler:
        np.save(file_handler, vectorized_corpus_matrix)

    return vectorized_corpus_matrix


def _transform_tfidf(data: [str], model: str, output: str) -> ndarray:
    """
    Transform data using tfidf

    Parameters
    ----------
    data : List of string
        List containing each corpus text file
        It's pre tokenized if the data given to _generate_tfidf is
    model : str
        Folder where the generated models will be allocated / loaded
    output : str
        Folder where vectorized corpus will be allocated

    Returns
    -------
    vectorized_corpus: ndarray of shape (n_samples, n_features)
        Vectorized corpus
    """
    # Cargamos el modelo
    with open(model + os.path.sep + 'tf-idf_vectorizer.joblib', 'rb') as model_file:
        model = load_model(model_file)

    # Predecimos los documentos para obtener los textos vectorizados
    vectorized_corpus_matrix = model.transform(data).toarray()

    # Guardamos el texto vectorizado
    with open(output + os.path.sep + 'tfidf_vectorized_corpus.npy', 'wb') as file_handler:
        np.save(file_handler, vectorized_corpus_matrix)

    return vectorized_corpus_matrix


def _transform_lda(data: [str], model: str, output: str) -> ndarray:
    """
    Transform data using lda

    Parameters
    ----------
    data : List of string
        List containing each corpus text file
        It's pre tokenized if the data given to _generate_lda is
    model : str
        Folder where the generated models will be allocated / loaded
    output : str
        Folder where vectorized corpus will be allocated

   Returns
    -------
    vectorized_corpus: ndarray of shape (n_samples, n_features)
        Vectorized corpus
    """

    with open(model + os.path.sep + 'BoW_vectorizer.joblib', 'rb') as model_file:
        bow_vectorizer = load_model(model_file)

    # Cargamos el modelo
    with open(model + os.path.sep + 'LDA_vectorizer.joblib', 'rb') as model_file:
        lda_vectorizer = load_model(model_file)

    data = bow_vectorizer.transform(data)

    # Predecimos los documentos para obtener los textos vectorizados
    vectorized_corpus_matrix = lda_vectorizer.transform(data)

    # Guardamos el texto vectorizado
    with open(output + os.path.sep + 'lda_vectorized_corpus.npy', 'wb') as file_handler:
        np.save(file_handler, vectorized_corpus_matrix)

    return vectorized_corpus_matrix


generate_vectorizer_methods = {
    'tf-idf': _generate_tfidf,
    'lda': _generate_lda
}

transform_methods = {
    'tf-idf': _transform_tfidf,
    'lda': _transform_lda
}


def vectorize_dataset(input: str, output: str, model: str, command: str, vectorizer: str, extension: str = '*.tok',
                      **kwargs) -> ndarray:
    """
    Vectorized the dataset given

    Parameters
    ----------
    input : str
        Input folder containing  text files
    output : str
        Folder where vectorized corpus will be allocated
    model : str
        Folder where the generated models will be allocated / loaded
    command : str
    vectorizer : str
        Indicates which algorithm will be used for vectorization
    extension : str
        Input file extension
    kwargs : dict
        Other parameters for the child functions.

    Returns
    -------
    vectorized_corpus: ndarray of shape (n_samples, n_features)
        Vectorized corpus
    """
    # Creamos las carpetas de output en caso de que no existan
    os.makedirs(output, exist_ok=True)
    os.makedirs(model, exist_ok=True)

    # Array donde irán todos los textos
    corpus = []

    # Buscamos los ficheros en la carpeta
    document_list = get_files(input + os.path.sep + '*' + extension)
    with open(output + os.path.sep + 'vectorized_files_path_mapping_list.json', 'w') as file:
        json.dump(document_list, file)

    for input_file_path in tqdm(document_list):
        with open(input_file_path, 'r', encoding='utf-8') as file_handler:
            file_text = file_handler.read()

        # Introducimos el texto a la lista
        corpus.append(file_text)

    if command is None:
        vectorized_corpus = transform_methods[vectorizer](corpus, model, output)
    else:
        vectorized_corpus = generate_vectorizer_methods[vectorizer](corpus, model, output, **kwargs)

    return vectorized_corpus


if __name__ == '__main__':
    def int_float(value):
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                raise argparse.ArgumentTypeError('The given value is neither an integer nor a float')
        return value


    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Argumentos genéricos
    parser.add_argument('-i', '--input', type=str, required=True, help='Input folder containing  text files.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Folder where vectorized corpus will be allocated.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Folder where the generated models will be allocated / loaded.')
    parser.add_argument('-v', '--vectorizer', type=str, choices=['tf-idf', 'lda'], required=True,
                        help='Indicates which algorithm will be used for vectorization.')
    parser.add_argument('-e', '--extension', type=str, default='*.tok', required=False, help='Input file extension.')
    subparsers = parser.add_subparsers(
        help='If no subcommand is given, the script will try to load an existing model from model folder.\n '
             'If a subcommand is given the script will train a new model depending of the subcommand.', dest='command')

    # Train lda arguments
    fit_lda_args = subparsers.add_parser('fit_transform_lda', help='Fit a new  LDA vectorizer from the given data')
    fit_lda_args.add_argument('--alpha', type=float, default=None, required=False, help='Alpha value for LDA')
    fit_lda_args.add_argument('--beta', type=float, default=None, required=False, help='Beta value for LDA')
    fit_lda_args.add_argument('--topics', type=int, default=200, required=False,
                              help='Number of features/topics/dimensions of the result vector.')
    fit_lda_args.add_argument('--iterations', type=int, default=10000, required=False, help='Max iterations for LDA')

    # Train Tf-Idf arguments
    fit_tfidf_args = subparsers.add_parser('fit_transform_tfidf', help='Fit a new Tf-Idf vectorizer from the given data')
    fit_tfidf_args.add_argument('--min_ngrams', type=int, default=1, required=False,
                                help='The lower boundary of the range of n-values for different n-grams to be extracted by Tf-Idf.')
    fit_tfidf_args.add_argument('--max_ngrams', type=int, default=1, required=False,
                                help='The  upper boundary of the range of n-values for different n-grams to be extracted by Tf-Idf.')
    fit_tfidf_args.add_argument('--min_df', type=int_float, default=1, required=False,
                                help='Ignore terms that have a document count/frequency strictly lower than the given threshold when building Tf-Idf.')
    fit_tfidf_args.add_argument('--max_df', type=int_float, default=0.9, required=False,
                                help='Ignore terms that have a document count/frequency strictly higher than the given threshold when building Tf-Idf.')
    fit_tfidf_args.add_argument('--max_features', type=int, default=None, required=False,
                                help='If not None, Tf-Idf builds a vocabulary that only consider the top max_features ordered by term frequency across the corpus.')
    fit_tfidf_args.add_argument('--sublinear_tf', type=bool, default=False, required=False,
                                help='Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).')

    # Parse args
    args = parser.parse_args()

    # Run the main function
    vectorize_dataset(**vars(args))
