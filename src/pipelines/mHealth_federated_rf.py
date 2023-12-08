import itertools
import multiprocessing
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join("..")))

from utils.mlflow_tracking_experiment import create_experiment


def start_server(number_round, number_clients, experiment_name):
    """
    Starts the server
    :param number_clients: Number of clients
    :param number_round: Number of rounds
    :param experiment_name: Name of the experiment
    """
    os.system(
        f"python server.py --number-of-clients {number_clients} --number-of-rounds {number_round} --experiment-name {experiment_name}"
    )


def start_client(dataset_path):
    """
    Starts a client
    :param dataset_path: Path to the dataset
    """
    print(f"Starting client for dataset {dataset_path}")
    os.system(
        f"python client.py --dataset-path {dataset_path} --client-id {dataset_path.split(os.sep)[-1].split('.')[0]}"
    )


if __name__ == "__main__":
    dataset_path = r"D:\Work\HAR-Comparison-Privacy\processed_data\activity_recognition_mHealth_less_classes_frequency_features"
    script_path = r"D:\Work\HAR-Comparison-Privacy\src\federated_rf"
    os.chdir(script_path)

    create_experiment("mHealth_federated_RF")

    number_rounds = [3, 5, 8, 10, 15, 20]
    number_clients = [6, 8, 10]
    combinations = list(itertools.product(number_rounds, number_clients))
    for number_round, number_client in combinations:
        print(
            f"Starting experiment with {number_round} rounds and {number_client} clients"
        )
        server_process = multiprocessing.Process(
            target=start_server,
            args=(number_round, number_client, "mHealth_federated_RF"),
        )
        server_process.start()
        time.sleep(15)

        with multiprocessing.Pool(number_client) as pool:
            csv_files = [
                os.path.join(dataset_path, data)
                for data in os.listdir(dataset_path)
                if data.endswith(".csv") and "all_users" not in data
            ]
            pool.map(start_client, csv_files)

        server_process.join()
    print("Done")
