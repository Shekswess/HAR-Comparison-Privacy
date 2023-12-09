import argparse
import os
import sys
from typing import Dict, List

import flwr as fl
from flwr.server.strategy import FedXgbBagging

sys.path.append(os.path.abspath(os.path.join("..")))

from utils.mlflow_tracking_experiment import (federated_log,
                                              set_default_tracking_uri,
                                              set_experiment)
from utils.vizualizations import plot_metrics

AVERAGE_F1_MACROS = []
AVERAGE_ACCURACIES = []
BEST_ROUND = 1
BEST_F1_MACRO = 0
BEST_ACCURACY = 0


def evaluate_metrics_aggregation(eval_metrics: List) -> Dict:
    """
    Aggregate evaluation metrics from all clients.
    :param eval_metrics: List of evaluation metrics from all clients.
    :return: Aggregated evaluation metrics.
    """
    global BEST_ROUND, BEST_F1_MACRO, BEST_ACCURACY
    total_num = sum([num for num, _ in eval_metrics])
    acc_aggregated = (
        sum([metric["accuracy"] * num for num, metric in eval_metrics]
            ) / total_num
    )
    f1_aggregated = (
        sum([metric["f1_macro"] * num for num, metric in eval_metrics]
            ) / total_num
    )
    metrics_aggregated = {
        "accuracy": acc_aggregated,
        "f1_macro": f1_aggregated}
    AVERAGE_F1_MACROS.append(f1_aggregated)
    AVERAGE_ACCURACIES.append(acc_aggregated)
    if f1_aggregated > BEST_F1_MACRO:
        BEST_F1_MACRO = f1_aggregated
        BEST_ACCURACY = acc_aggregated
        BEST_ROUND = len(AVERAGE_F1_MACROS)
    return metrics_aggregated


if __name__ == "__main__":
    base_figure_path = r"..\figures\federated_xgboost"
    os.makedirs(base_figure_path, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number-of-clients", type=int,
        help="Number of clients.", required=True
    )
    parser.add_argument(
        "--number-of-rounds", type=int,
        help="Number of rounds.", required=True
    )
    parser.add_argument(
        "--experiment-name", type=str,
        help="Name of the experiment.", required=True
    )
    num_clients = parser.parse_args().number_of_clients
    num_rounds_global = parser.parse_args().number_of_rounds
    experiment_name = parser.parse_args().experiment_name

    set_default_tracking_uri(
        "file:////D:\Work\HAR-Comparison-Privacy\src\pipelines\mlruns"
        )
    set_experiment(experiment_name)

    pool_size = num_clients
    num_rounds = num_rounds_global
    num_clients_per_round = num_clients
    num_evaluate_clients = num_clients

    strategy = FedXgbBagging(
        fraction_fit=(float(num_clients_per_round) / pool_size),
        min_fit_clients=num_clients_per_round,
        min_available_clients=pool_size,
        min_evaluate_clients=num_evaluate_clients,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    print("Starting Flower server...")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    run_name = f"xgboost_federated_{num_clients}_clients_{num_rounds}_rounds"
    average_f1 = sum(AVERAGE_F1_MACROS) / len(AVERAGE_F1_MACROS)
    average_acc = sum(AVERAGE_ACCURACIES) / len(AVERAGE_ACCURACIES)
    plot_metrics(
        AVERAGE_F1_MACROS,
        "F1 Macro",
        "F1 Macro per round",
        os.path.join(base_figure_path, f"{run_name}_f1_macro.png"),
    )
    plot_metrics(
        AVERAGE_ACCURACIES,
        "Accuracy",
        "Accuracy per round",
        os.path.join(base_figure_path, f"{run_name}_accuracy.png"),
    )
    federated_log(
        run_name,
        average_acc,
        average_f1,
        BEST_ACCURACY,
        BEST_F1_MACRO,
        BEST_ROUND,
        AVERAGE_ACCURACIES,
        AVERAGE_F1_MACROS,
        os.path.join(base_figure_path, f"{run_name}_f1_macro.png"),
    )
