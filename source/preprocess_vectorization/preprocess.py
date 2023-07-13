import argparse
import re
import string
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import spacy
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

punctuation_regex = re.compile('[' + re.escape(string.punctuation) + ']')


def remove_accents_punctuation_stopwords(text: str) -> str:
    """
    Remove accents, punctuation and stopwords from pStr

    Parameters
    ----------
    text : str
        A text file to preprocess
        It is recommended that you tokenize and
        lemmatize it before runing this function

    Returns
    -------
    String without accents, punctuation or stopwords
    """

    stpWords = set(stopwords.words('spanish'))  # Definición de los stopwords

    cleanStr = ' ' + text.lower() + ' '

    # Se eliminan comas y otros
    cleanStr = punctuation_regex.sub(' ', cleanStr)
    cleanStr = re.sub(' +', ' ', cleanStr)

    # Se eliminan acentos y otros carácteres raros
    cleanStr = re.sub(r'([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+', r'\1',
                      unicodedata.normalize('NFD', cleanStr), 0, re.I)
    cleanStr = unicodedata.normalize('NFC', cleanStr)

    # Se eliminan las stopwords
    for word in stpWords:
        word = re.sub(r'([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+', r'\1',
                      unicodedata.normalize('NFD', word), 0, re.I)
        word = unicodedata.normalize('NFC', word)

        cleanStr = re.sub(r'[ \t\n]' + word + r'[ \t\n]', ' ', cleanStr)

    return cleanStr.strip()


def tokenize_lemmatize_text(text: str) -> str:
    """
    Tokenize and lemmatize the text given

    Parameters
    ----------
    text : str
        A text file to preprocess

    Returns
    -------
    tokenized_text : str
        Tokenied and lemmatized text
    """
    # Tokenizamos
    tokenized_text = ' '.join(word_tokenize(text, "spanish"))

    # Lematizamos
    tokenized_text = spacy.load('es_core_news_sm')(tokenized_text)
    tokenized_text = ' '.join(token.lemma_.lower() for token in tokenized_text)

    return tokenized_text


def tokenize_clean_text(input_text: str) -> str:
    """
    Tokenize, lemmatize and clean the text in the path given and
    saves the result in the output given path.

    Parameters
    ----------
    input_text : String
        Text to tokenize
    """

    # Tokenizamos y lematizamos el texto y limpiamos los stop words y tildes del texto tokenizado
    file_text = tokenize_lemmatize_text(input_text)
    file_text = remove_accents_punctuation_stopwords(file_text)
    return file_text


def tokenize_clean_set(input_path: str, output_path: str = None, text_column: str = 'Text'):
    """
    Tokenize, lemmatize and clean the texts that matches the extension in the path given.

    Text will be saved in the output path. This function uses multithreading.

    Parameters
    ----------
    input_path : String
        Input csv file
    output_path : String
        Output csv file, if not set defaults to input file
    text_column : String
        Name of the data column containing the texts
    max_tasks_per_child: int
        Reduce this number in case you get a Memory Insufficient Error.
    """

    if not output_path:
        output_path = input_path

    # Creamos la carpeta de output en caso de que no exista
    Path(output_path).parent.mkdir(exist_ok=True)

    # Cargamos el fichero:
    data = pd.read_csv(input_path)

    # Obtenemos los textos:
    texts = data[text_column]

    # Buscamos los ficheros en la carpeta y empleamos multiprocessing para procesarlos
    multithread_task = []
    tokenized_texts = [None] * len(texts)

    ##############################################################################
    # ------ IMPORTANT
    # Reduce maxtasksperchild in case you get a Memory Insufficient Error.
    ##############################################################################
    with Pool(maxtasksperchild=10) as pool:
        spacy.require_gpu()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError
            multithread_task.append(pool.apply_async(func=tokenize_clean_text, args=(text,)))

        for indx, task in enumerate(tqdm(multithread_task)):
            tokenized_texts[indx] = task.get()

    data[f'{text_column}_Tokenized'] = pd.Series(tokenized_texts)

    data.to_csv(output_path, index=False)


if __name__ == '__main__':
    """"
    Recorre los ficheros de data/raw
    Los tokeniza palabra por palabra
    Almacena los fichero en data/tokenized

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input csv file.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output csv file, if not set defaults to input file.')
    parser.add_argument('-c', '--text_column', type=str, default='Text', required=False, help="Name of the data column containing the texts.")
    args = parser.parse_args()

    tokenize_clean_set(args.input, args.output, args.text_column)
