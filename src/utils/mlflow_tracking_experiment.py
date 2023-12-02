from typing import List, Union

import mlflow as ml
import numpy as np


def create_experiment(experiment_name: str):
    """
    Create an experiment in MLFlow
    :param experiment_name: Name of the experiment
    """
    ml.create_experiment(experiment_name)


def set_experiment(experiment_name: str):
    """
    Set an experiment in MLFlow
    :param experiment_name: Name of the experiment
    """
    ml.set_experiment(experiment_name)


def _log_confusion_matrix(confusion_matrix_path: str):
    """
    Log the confusion matrix in MLFlow
    :param confusion_matrix_path: Path of the confusion matrix
    """
    ml.log_artifact(confusion_matrix_path)


def _log_test_n_pred(test: Union[np.ndarray, List],
                     pred: Union[np.ndarray, List]):
    """
    Log the test and pred values in MLFlow
    :param test: List of test values
    :param pred: List of pred values
    """
    ml.log_param("Test-Values", test)
    ml.log_param("Pred-Values", pred)


def _log_accuracy_n_f1(accuracy: float, f1: float):
    """
    Log the accuracy and f1 values in MLFlow
    :param accuracy: Accuracy value
    :param f1: F1 value
    """
    ml.log_metric("Accuracy", accuracy)
    ml.log_metric("F1-Macro", f1)


def _log_subjects(train_subjects: List, test_subjects: List):
    """
    Log the train and test subjects in MLFlow
    :param train_subjects: List of train subjects
    :param test_subjects: List of test subjects
    """
    ml.log_param("Train-Subjects", train_subjects)
    ml.log_param("Test-Subjects", test_subjects)


def train_test_log(
    run_name: str,
    test: List,
    pred: List,
    accuracy: float,
    f1: float,
    confusion_matrix_path: str,
    train_subjects: List,
    test_subjects: List,
):
    """
    Log the test, pred, accuracy, f1, confusion matrix,
    train and test subjects in MLFlow
    :param test: List of test values
    :param pred: List of pred values
    :param accuracy: Accuracy value
    :param f1: F1 value
    :param confusion_matrix_path: Path of the confusion matrix
    """
    ml.start_run(run_name=run_name)
    _log_subjects(train_subjects, test_subjects)
    _log_test_n_pred(test, pred)
    _log_accuracy_n_f1(accuracy, f1)
    _log_confusion_matrix(confusion_matrix_path)
    ml.end_run()
