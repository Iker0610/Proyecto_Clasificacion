import argparse
from pathlib import Path

import pandas as pd

classes = ["Description", "Sintomas", "Causas", "FactoresRiesgo", "Complicaciones", "Prevencion", "Diagnostico", "Tratamiento"]
metric_columns = ['loss', 'categorical_accuracy', 'weighted_f1', 'micro_f1', 'macro_f1'] + [f'{metric} {c}' for metric in ['Recall', 'Precision', 'F-Score'] for c in classes]


def main(input_csv, output_csv):
    # Create output file
    Path(output_csv).parent.mkdir(exist_ok=True)

    # Load the CSV
    df = pd.read_csv(input_csv)

    # Get metrics
    metrics = df[['trial_id'] + metric_columns].copy()

    # Delete from original
    df.drop(columns=metric_columns, inplace=True)

    # Transform metric columns to dataframe (one metric per line)
    lines = []
    for row in metrics.iloc:
        row_id = row['trial_id']
        for metric in metric_columns[0:2]:
            lines.append([row_id, metric, None, row[metric]])
        for metric in metric_columns[2:5]:
            metric_type = metric.split('_')[0].capitalize()
            lines.append([row_id, 'F-Score', metric_type, row[metric]])
        for metric in metric_columns[5:]:
            metric_name, metric_class = metric.split()
            lines.append([row_id, metric_name, metric_class, row[metric]])
    metrics = pd.DataFrame(lines, columns=['trial_id', 'metric', 'metric type/class', 'metric value'])

    # Merge with original (to keep other hiper params data)
    result = pd.merge(df, metrics, on='trial_id')

    # Save
    result.to_csv(output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_csv', type=str, required=True, help='Input CSV with all data.')
    parser.add_argument('-o', '--output_csv', type=str, required=True, help='Output CSV with the metrics.')
    args = parser.parse_args()
    main(**vars(args))
