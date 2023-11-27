import sys
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from utils.validation import subject_cross_validation


if __name__ == "__main__":
    path = r'D:\Work\HAR-Comparison-Privacy\processed_data\activity_recognition_wisdm\all_users.csv'
    data = pd.read_csv(path)
    subject_column = "User_ID"
    label_column = "Label"
    algo_type = "RandomForest"

    global_test, global_pred, global_accuracy, global_auc, global_f1, global_report, mean_accuracy, mean_f1 = subject_cross_validation(data, subject_column, label_column, algo_type)
