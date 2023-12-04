from typing import List, Union

import mlflow as ml
import numpy as np


def set_default_tracking_uri(tracking_uri: str):
    """
    Set the default tracking uri
    :param tracking_uri: Tracking uri
    """
    ml.set_tracking_uri(tracking_uri)


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


def _log_test_subjects(test_subjects: List):
    """
    Log the test subjects in MLFlow
    :param test_subjects: List of test subjects
    """
    ml.log_param("Test-Subjects", test_subjects)


def _log_accuracy_n_f1_list(accuracy_list: List[float], f1_list: List[float]):
    """
    Log the accuracy and f1 values in MLFlow
    :param accuracy: Accuracy value
    :param f1: F1 value
    """
    ml.log_param("Accuracy-List", accuracy_list)
    ml.log_param("F1-Macro-List", f1_list)


def _log_average_accuracy_n_f1_list(
    average_accuracy_list: List[float], average_f1_list: List[float]
):
    """
    Log the average accuracy and f1 values in MLFlow
    :param average_accuracy: Average accuracy value
    :param average_f1: Average F1 value
    """
    ml.log_param("Average-Accuracy-List", average_accuracy_list)
    ml.log_param("Average-F1-Macro-List", average_f1_list)


def _log_average_accuracy_n_f1(average_accuracy: float, average_f1: float):
    """
    Log the average accuracy and f1 values in MLFlow
    :param average_accuracy: Average accuracy value
    :param average_f1: Average F1 value
    """
    ml.log_metric("Average-Accuracy", average_accuracy)
    ml.log_metric("Average-F1-Macro", average_f1)


def _log_best_accuracy_n_f1(best_accuracy: float,
                            best_f1: float, best_round: int):
    """
    Log the best accuracy and f1 values in MLFlow
    :param best_accuracy: Best accuracy value
    :param best_f1: Best F1 value
    """
    ml.log_metric("Best-Accuracy", best_accuracy)
    ml.log_metric("Best-F1-Macro", best_f1)
    ml.log_metric("Best-Round", best_round)


def _log_plot_metrics(figure_path: str):
    """
    Log the plot metrics in MLFlow
    :param figure_path: Path of the plot metrics
    """
    ml.log_artifact(figure_path)


def federated_log(
    run_name: str,
    average_accuracy: float,
    average_f1: float,
    best_accuracy: float,
    best_f1: float,
    best_round: int,
    average_accuracy_list: List[float],
    average_f1_list: List[float],
    figure_path: str,
):
    """
    Log the average accuracy, average f1, best accuracy, best f1, best round,
    average accuracy list, average f1 list, figure path in MLFlow
    :param average_accuracy: Average accuracy value
    :param average_f1: Average F1 value
    :param best_accuracy: Best accuracy value
    :param best_f1: Best F1 value
    :param best_round: Best round value
    :param average_accuracy_list: Average accuracy list
    :param average_f1_list: Average F1 list
    :param figure_path: Path of the figure
    """
    ml.start_run(run_name=run_name)
    _log_average_accuracy_n_f1(average_accuracy, average_f1)
    _log_best_accuracy_n_f1(best_accuracy, best_f1, best_round)
    _log_average_accuracy_n_f1_list(average_accuracy_list, average_f1_list)
    _log_plot_metrics(figure_path)
    ml.end_run()


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


def loso_log(
    run_name: str,
    test: List,
    pred: List,
    accuracies: List[float],
    f1s: List[float],
    test_subjects: List,
    average_accuracy: float,
    average_f1: float,
):
    """
    Log the test, pred, accuracies, f1s, test subjects,
    average accuracy and average f1 in MLFlow
    :param test: List of test values
    :param pred: List of pred values
    :param accuracies: List of accuracies
    :param f1s: List of f1s
    :param test_subjects: List of test subjects
    :param average_accuracy: Average accuracy value
    :param average_f1: Average F1 value
    """
    ml.start_run(run_name=run_name)
    _log_test_n_pred(test, pred)
    _log_average_accuracy_n_f1(average_accuracy, average_f1)
    _log_accuracy_n_f1_list(accuracies, f1s)
    _log_test_subjects(test_subjects)
    ml.end_run()
