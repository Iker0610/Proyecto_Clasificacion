# Código original obtenido de StackOverflow, después adaptado para añadir funcionalidades
# Usuario: stackoverflowuser2010
# Link: https://stackoverflow.com/questions/50781562/stratified-splitting-of-pandas-dataframe-into-training-validation-and-test-set
import argparse
import os
from math import fsum
from typing import Union

from pandas import DataFrame
from sklearn.model_selection import train_test_split


def split_stratified_into_train_val_test(df_input: Union[str, DataFrame], stratify_colname: str = 'label', frac_train: float = 0.6, frac_val: float = 0.25, frac_test: float = 0.15, output_folder: str = None, output_file_base_name: str = None, random_state: int = None):
    """
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe or str
        Input dataframe to be split. Or path to the dataframe csv
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
        The ratio of the dataframe will be split into validation/dev data.
        The value should be expressed as float fractions and should sum to 1.0 with the others.
    frac_val   : float
        The ratio of the dataframe will be split into validation/dev data.
        The value should be expressed as float fractions and should sum to 1.0 with the others.
    frac_test  : float
        The ratio of the dataframe will be split into test data.
        The value should be expressed as float fractions and should sum to 1.0 with the others.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().
    output_folder: str
        Optional output folder where partitions will be saved.
    output_file_base_name: str
        Output files' base name. Example: 'dataset', then the next files will be created: dataset.train.csv, dataset.dev.csv, dataset.test.csv

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    """

    if fsum([frac_train, frac_val, frac_test]) != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError(f'{stratify_colname} is not a column in the dataframe')

    X = df_input  # Contains all columns.
    y = df_input[[stratify_colname]]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X, y, stratify=y,
                                                          train_size=frac_train,
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / fsum([frac_val, frac_test])
    df_val, df_test, y_val, y_test = train_test_split(df_temp, y_temp, stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    if output_folder and output_file_base_name:
        os.makedirs(output_folder, exist_ok=True)
        df_train.to_csv(f'{output_folder}/{output_file_base_name}.train.csv', index=False)
        df_val.to_csv(f'{output_folder}/{output_file_base_name}.dev.csv', index=False)
        df_test.to_csv(f'{output_folder}/{output_file_base_name}.test.csv', index=False)

    return df_train, df_val, df_test


if __name__ == '__main__':
    # TODO: Hacer todo el argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--df_input', type=str, required=True, help='Input dataframe to be split. Or path to the dataframe csv.')
    parser.add_argument('-c', '--stratify_colname', type=str, required=False, default='label', help='The name of the column that will be used for stratification.')
    parser.add_argument('--frac_train', type=float, required=False, default=0.6, help='The ratio of the dataframe will be split into train.')
    parser.add_argument('--frac_val', type=float, required=False, default=0.25, help='The ratio of the dataframe will be split into validation.')
    parser.add_argument('--frac_test', type=float, required=False, default=0.15, help='The ratio of the dataframe will be split into test.')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Optional output folder where partitions will be saved.')
    parser.add_argument('-n', '--output_file_base_name', type=str, required=True, help="Output files' base name. Example: 'dataset', then the next files will be created: dataset.train.csv, dataset.dev.csv, dataset.test.csv")
    parser.add_argument('-r', '--random_state', type=int, required=False, default=None, help='Value to be passed to train_test_split().')

    args = parser.parse_args()

    split_stratified_into_train_val_test(**vars(args))
