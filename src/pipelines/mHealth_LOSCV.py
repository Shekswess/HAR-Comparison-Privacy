import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join("..")))
from tqdm import tqdm

from utils.mlflow_tracking_experiment import create_experiment, set_experiment, loso_log
from utils.validation import leave_one_subject_out_validation

if __name__ == "__main__":
    path ="/home/bojan-emteq/Work/HAR-Comparison-Privacy/processed_data/activity_recognition_mHealth_less_classes_frequency_features/all_users.csv"
    data = pd.read_csv(path)
    subject_column = "User_ID"
    label_column = "Label"
    labels = {1: "Still", 2: "Walking", 3: "Stairs", 4: "Jogging"}
    create_experiment("mHealth_less_classes_frequency_features_LOSCV")
    set_experiment("mHealth_less_classes_frequency_features_LOSCV")
    algo_types = ["XGBoost", "LightGBM", "RandomForest"]
    for algo_type in tqdm(algo_types, desc="Algo Types", total=len(algo_types)):
        print("\nAlgo Type: ", algo_type)
        (
            accuracy,
            f1,
            test_subjects,
            mean_accuracy,
            mean_f1,
        ) = leave_one_subject_out_validation(
            data, subject_column, label_column, algo_type
        )
        loso_log(
            algo_type, accuracy, f1, test_subjects, mean_accuracy, mean_f1
        )
