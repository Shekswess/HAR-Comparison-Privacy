import argparse
import warnings
from logging import INFO
from typing import Union

import flwr as fl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class RfClient(fl.client.Client):
    def __init__(self, x_train, y_train, x_test, y_test, params, num_train, num_test):
        """
        Random forest client.
        :param x_train: The training dataset.
        :param y_train: The training labels.
        :param x_test: The test dataset.
        :param y_test: The test labels.
        :param params: The parameters for training.
        :param num_train: The number of training examples.
        :param num_test: The number of test examples.
        """
        self.rf = None
        self.config = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.params = params
        self.num_train = num_train
        self.num_test = num_test

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Return the local model parameters.
        :param ins: GetParametersIns
        :return: GetParametersRes
        """
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """
        Train the model on the local dataset.
        :param ins: FitIns
        :return: FitRes
        """
        if self.rf is None:
            self.rf = RandomForestClassifier(**self.params)
            self.rf.fit(self.x_train, self.y_train)
            self.config = ins.config
        else:
            self.rf.fit(self.x_train, self.y_train)
            self.config = ins.config

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            num_examples=self.num_train,
            parameters=Parameters(tensor_type="", tensors=[]),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate the model on the local dataset.
        :param ins: EvaluateIns
        :return: EvaluateRes
        """

        y_pred = self.rf.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="macro")
        results = {"Client ID": client_id, "Accuracy": accuracy, "F1": f1}
        print(results)

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            num_examples=self.num_test,
            loss=0.0,
            metrics={"accuracy": accuracy, "f1_macro": f1},
        )


def train_test_split_data(
    data: pd.DataFrame, test_fraction: float, seed: int
) -> Union[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test set by given split rate
    and each label to be balanced.
    :param data: the dataset to be split
    :param test_fraction: the split rate
    :param seed: random seed
    :return: the train and test set
    """
    labels = data["Label"].unique()
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for label in labels:
        data_label = data[data["Label"] == label]
        data_label_train, data_label_test = train_test_split(
            data_label, test_size=test_fraction, random_state=seed
        )
        train_data = pd.concat([train_data, data_label_train])
        test_data = pd.concat([test_data, data_label_test])

    return train_data, test_data


def transform_dataset_to_x_y(data: pd.DataFrame):
    """
    Transform the dataset into DMatrix for xgboost.
    :param data: the dataset to be transformed
    :return: the transformed dataset in DMatrix format
    """
    x_data = data.drop(["User_ID", "Label"], axis=1)
    y_data = data["Label"]
    return x_data, y_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=str, help="Path to the dataset.", required=True
    )
    parser.add_argument("--client-id", type=str, help="Client ID.", required=True)
    args = parser.parse_args()

    client_id = args.client_id
    dataset_path = args.dataset_path
    df = pd.read_csv(dataset_path)

    train_data, test_data = train_test_split_data(df, test_fraction=0.2, seed=42)

    x_train, y_train = transform_dataset_to_x_y(train_data)
    x_test, y_test = transform_dataset_to_x_y(test_data)

    params = {
        "n_estimators": 50,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": None,
    }

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=RfClient(x_train, y_train, x_test, y_test, params, num_train, num_test),
    )
