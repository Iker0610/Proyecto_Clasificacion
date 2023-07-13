import argparse
import json
from pathlib import Path
from glob import glob as get_files

import pandas as pd

useless_columns = [
    "experiment_id",
    "time_this_iter_s",
    "training_iteration",
    "date",
    "hostname",
    "node_ip",
    "timestamp",
    "done",
    "timesteps_total",
    "episodes_total",
    "pid",
    "time_since_restore",
    "timesteps_since_restore",
    "iterations_since_restore"
]

dataset_map = {'lda_dataset': 'LDA', 'doc2vec_all_corpus_dataset': 'Doc2Vec_Big', 'doc2vec_dataset': 'Doc2Vec_Little',
               'LDA': 'LDA', 'Doc2Vec_Big': 'Doc2Vec_Big', 'Doc2Vec_Little': 'Doc2Vec_Little'}

class_label_mapping = {
    0: "Description",
    1: "Sintomas",
    2: "Causas",
    3: "FactoresRiesgo",
    4: "Complicaciones",
    5: "Prevencion",
    6: "Diagnostico",
    7: "Tratamiento",
    "Description": 0,
    "Sintomas": 1,
    "Causas": 2,
    "FactoresRiesgo": 3,
    "Complicaciones": 4,
    "Prevencion": 5,
    "Diagnostico": 6,
    "Tratamiento": 7
}


def fscore(p, r):
    return 2 * p * r / (p + r)


def get_results_json(input_folder):
    """
    Precondition: The script expects an independent JSON for each line of the file. And also a config key with a dictionary inside each json

    Valid file example:
    {"field1_1": "value1_1", "field1_2": "value1_2", "field1_3": "value1_3"}
    {"field2_1": "value2_1", "field2_3": "value2_2", "field2_3": "value2_3"}

    Invalid file example:
    {"field1_1": "value1_1",
     "field1_2": "value1_2",
     "field1_3": "value1_3"}
    """
    results = []
    for file in get_files(f'{input_folder}/*/result.json'):
        with open(file, encoding='utf8') as f:
            for text in f.read().splitlines():
                if text:
                    result = json.loads(text)
                    # Flatten the dictionary
                    config = result.pop('config')
                    result.update(config)
                    results.append(result)

    return results


def process_df(df):
    # Clear columns
    df = df.drop(columns=useless_columns)

    # Adjust dataset names
    df['dataset'] = df['dataset'].map(dataset_map)

    # Get F-Score for each class
    for i in range(8):
        df[f'F-Score {class_label_mapping[i]}'] = fscore(df[f'Precision_{i}'], df[f'Recall_{i}'])
        df[f'F-Score {class_label_mapping[i]}'].fillna(0, inplace=True)

    # Rename Precision and Recall columns
    df = df.rename(columns={f'Precision_{i}': f'Precision {class_label_mapping[i]}' for i in range(8)})
    df = df.rename(columns={f'Recall_{i}': f'Recall {class_label_mapping[i]}' for i in range(8)})

    return df


def main(input_folder: str, output_folder: str, file_name: str):
    # Create output folder
    Path(output_folder).mkdir(exist_ok=True)

    # Unify all the experiments in a json list
    results_json = get_results_json(input_folder)

    # Save the list
    with open(f'{output_folder}/{file_name}.json', 'w', encoding='utf8') as f:
        json.dump(results_json, f)

    # Convert it to a dataframe
    df = pd.DataFrame(results_json)

    # Process and save dataframe
    df = process_df(df)
    df.to_csv(f'{output_folder}/{file_name}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Ray-Tune experiment folder.')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Output folder.')
    parser.add_argument('-n', '--file_name', type=str, required=True, help='Outfiles\' name.')
    args = parser.parse_args()
    main(**vars(args))
