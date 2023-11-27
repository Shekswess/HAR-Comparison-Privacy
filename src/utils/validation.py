import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from xgboost import XGBClassifier
from tqdm import tqdm


def subject_cross_validation(
    data: pd.DataFrame, subject_column: str, label_column: str, algo_type: str
):
    if algo_type == "RandomForest":
        algo = RandomForestClassifier()
    elif algo_type == "XGBoost":
        algo = XGBClassifier()
    elif algo_type == "LightGBM":
        algo = LGBMClassifier()
    else:
        raise ValueError("Invalid algo_type")

    global_test = []
    global_pred = []
    global_accuracy = []
    global_auc = []
    global_f1 = []
    global_report = []

    mean_accuracy = []
    mean_f1 = []

    subjects = [1,2]
    for subject in tqdm(subjects, desc="Subject Cross Validation"):
        X_train = data[data[subject_column] != subject].drop(
            [label_column, subject_column], axis=1
        )
        y_train = data[data[subject_column] != subject][label_column]
        X_test = data[data[subject_column] == subject].drop(
            [label_column, subject_column], axis=1
        )
        y_test = data[data[subject_column] == subject][label_column]

        algo.fit(X_train, y_train)

        y_pred = algo.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred)

        global_test.append({"subject": subject, "test": y_test})
        global_pred.append({"subject": subject, "pred": y_pred})
        global_accuracy.append({"subject": subject, "accuracy": accuracy})
        global_f1.append({"subject": subject, "f1": f1})
        global_report.append({"subject": subject, "report": report})

        mean_accuracy.append(accuracy)
        mean_f1.append(f1)

    mean_accuracy = sum(mean_accuracy) / len(mean_accuracy)
    mean_f1 = sum(mean_f1) / len(mean_f1)

    return (
        global_test,
        global_pred,
        global_accuracy,
        global_auc,
        global_f1,
        global_report,
        mean_accuracy,
        mean_f1,
    )
