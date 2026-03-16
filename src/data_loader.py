import os
import pandas as pd
import wfdb
import numpy as np

def load_metadata(dataset_path):
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    return pd.read_csv(metadata_path)


def load_record(dataset_path, patient_id):
    record_path = os.path.join(
        dataset_path,
        "files",
        str(patient_id),
        str(patient_id)
    )

    record = wfdb.rdrecord(record_path)
    signal = record.p_signal

    return signal


def load_dataset(dataset_path):
    metadata = load_metadata(dataset_path)

    X = []
    y = []

    for _, row in metadata.iterrows():
        patient_id = row["patient_id"]
        label = row["brugada"]

        signal = load_record(dataset_path, patient_id)

        X.append(signal)
        y.append(label)

    return np.array(X), np.array(y)