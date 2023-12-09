import random
from typing import List, Union

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def train_test_validation(
    data: pd.DataFrame,
    subject_column: str,
    label_column: str,
    algo_type: str,
    train_ratio: float = 0.8,
) -> Union[List, List, float, float, List, List]:
    """
    Perform train test validation
    :param data: Dataframe
    :param subject_column: Column name of the subject
    :param label_column: Column name of the label
    :param algo_type: Type of algorithm
    :param train_ratio: Ratio of train data
    :return: List of test, pred, accuracy, auc, f1, train&test_subjects
    """
    if algo_type == "RandomForest":
        algo = RandomForestClassifier()
    elif algo_type == "XGBoost":
        algo = XGBClassifier()
    elif algo_type == "LightGBM":
        algo = LGBMClassifier()
    else:
        raise ValueError("Invalid algo_type")

    subjects = data[subject_column].unique()
    random.seed(42)
    train_subjects = random.sample(list(subjects),
                                   int(len(subjects) * train_ratio))
    test_subjects = [subject for subject in subjects
                     if subject not in train_subjects]

    label_encoder = LabelEncoder()

    if data.isnull().values.any():
        data = data.dropna()

    X_train = data[data[subject_column].isin(train_subjects)].drop(
        [label_column, subject_column], axis=1
    )
    y_train = data[data[subject_column].isin(train_subjects)][label_column]
    y_train = label_encoder.fit_transform(y_train)

    X_test = data[data[subject_column].isin(test_subjects)].drop(
        [label_column, subject_column], axis=1
    )
    y_test = data[data[subject_column].isin(test_subjects)][label_column]

    algo.fit(X_train, y_train)

    y_pred = algo.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    y_test = y_test.to_numpy()

    print(f"Train Subjects: {train_subjects}")
    print(f"Test Subjects: {test_subjects}")
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")

    return y_test, y_pred, accuracy, f1, train_subjects, test_subjects


def leave_one_subject_out_validation(
    data: pd.DataFrame, subject_column: str, label_column: str, algo_type: str
) -> Union[List, List, List, float, float]:
    """
    Perform leave one subject out validation
    :param data: Dataframe
    :param subject_column: Column name of the subject
    :param label_column: Column name of the label
    :param algo_type: Type of algorithm
    :return: List of test, pred, accuracy, auc, f1, test_subjects,
    mean_accuracy, mean_f1
    """
    if algo_type == "RandomForest":
        algo = RandomForestClassifier()
    elif algo_type == "XGBoost":
        algo = XGBClassifier()
    elif algo_type == "LightGBM":
        algo = LGBMClassifier()
    else:
        raise ValueError("Invalid algo_type")

    subjects = data[subject_column].unique()
    label_encoder = LabelEncoder()

    if data.isnull().values.any():
        data = data.dropna()

    accuracy_list = []
    f1_list = []
    test_subjects_list = []

    for subject in subjects:
        print(f"Subject: {subject}")
        train_subjects = [subject_ for subject_ in subjects
                          if subject_ != subject]
        test_subjects = [subject]

        X_train = data[data[subject_column].isin(train_subjects)].drop(
            [label_column, subject_column], axis=1
        )
        y_train = data[data[subject_column].isin(train_subjects)][label_column]
        y_train = label_encoder.fit_transform(y_train)

        X_test = data[data[subject_column].isin(test_subjects)].drop(
            [label_column, subject_column], axis=1
        )
        y_test = data[data[subject_column].isin(test_subjects)][label_column]

        algo.fit(X_train, y_train)

        y_pred = algo.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        accuracy_list.append(accuracy)
        f1_list.append(f1)
        test_subjects_list.append(test_subjects)

    mean_accuracy = sum(accuracy_list) / len(accuracy_list)
    mean_f1 = sum(f1_list) / len(f1_list)

    return (
        accuracy_list,
        f1_list,
        test_subjects_list,
        mean_accuracy,
        mean_f1,
    )
