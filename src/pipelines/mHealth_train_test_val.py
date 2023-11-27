import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join("..")))
from tqdm import tqdm

from utils.mlflow_tracking_experiment import (
    create_experiment,
    set_experiment,
    train_test_log,
)
from utils.validation import train_test_validation
from utils.vizualizations import confusion_matrix_heatmap

if __name__ == "__main__":
    path = r"D:\Work\HAR-Comparison-Privacy\processed_data\activity_recognition_mHealth\all_users.csv"
    base_figure_path = r"..\figures\mHealth_80_20"
    if not os.path.exists(base_figure_path):
        os.makedirs(base_figure_path)
    data = pd.read_csv(path)
    subject_column = "User_ID"
    label_column = "Label"
    labels = data[label_column].unique()
    create_experiment("mHealth_80_20")
    set_experiment("mHealth_80_20")
    algo_types = ["RandomForest", "XGBoost", "LightGBM"]
    for algo_type in tqdm(algo_types, desc="Algo Types", total=len(algo_types)):
        print("\nAlgo Type: ", algo_type)
        y_test, y_pred, accuracy, f1, train_sub, test_sub = train_test_validation(
            data, subject_column, label_column, algo_type
        )
        confusion_matrix_path = os.path.join(
            base_figure_path, f"{algo_type}_confusion_matrix.png"
        )
        confusion_matrix_heatmap(
            y_test,
            y_pred,
            labels,
            algo_type,
            confusion_matrix_path,
        )
        train_test_log(
            algo_type,
            y_test,
            y_pred,
            accuracy,
            f1,
            confusion_matrix_path,
            train_sub,
            test_sub,
        )
