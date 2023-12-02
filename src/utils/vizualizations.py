from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def confusion_matrix_heatmap(
    y_test: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    labels: List[str],
    title: str,
    save_path: str,
):
    """
    Plot the confusion matrix heatmap
    :param y_test: List of test values
    :param y_pred: List of pred values
    :param labels: Dict of labels
    :param title: Title of the plot
    :param save_path: Path to save the plot
    """
    labels_values = list(labels.values())
    y_test = [labels[x] for x in y_test]
    y_pred = [labels[x] for x in y_pred]
    cm = confusion_matrix(y_test, y_pred, labels=labels_values)
    cm_df = pd.DataFrame(cm, index=labels_values, columns=labels_values)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="g")
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(save_path)
