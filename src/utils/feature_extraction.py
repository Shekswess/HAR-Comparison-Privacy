from typing import List, Union

import numpy as np
import pandas as pd
from scipy import stats


def _moving_window(
    data: Union[np.ndarray, pd.Series], window: float, step: float
) -> np.ndarray:
    """
    Segment data into overlapping windows(if the window and step are equal,
    there is no overlap)
    :param data: data to segment
    :param window: window length
    :param step: window step
    :return: segmented data
    """
    if type(data) is list:
        data = np.array(data)
    if type(data) is pd.core.frame.DataFrame:
        data = data.values.copy()
        if data.shape[1] > 1:
            raise ValueError("Expected only one column of data.")
    if type(data) is pd.core.frame.Series:
        data = data.values.copy()
    if data.shape[0] < window:
        raise ValueError("Input array is shorter than the specified window.")
    one_column = np.array([]).reshape((0, window))
    data_flat = data.flatten()

    num_windows = int((len(data_flat) - window) / step + 1)
    indexer = np.arange(window)[None, :] + step * np.arange(num_windows)[:, None]
    slide = data_flat[indexer]
    one_column = np.concatenate((one_column, slide))
    one_column.astype("float32")
    return one_column


def generate_labels(
    data: Union[np.ndarray, pd.Series],
    label_column: str,
    win_length: float,
    overlap: float,
) -> np.ndarray:
    """
    Generate labels for the segmented data
    :param data: data to segment
    :param label_column: column containing the labels
    :param win_length: window length
    :param overlap: window overlap
    :return: labels for the segmented data
    """
    labels = _moving_window(data[label_column], win_length, overlap)
    labels = np.apply_along_axis(lambda x: stats.mode(x).mode, 1, labels)
    return labels


def _calculate_statistical_features(
    data: Union[np.ndarray, pd.Series], column: str
) -> pd.DataFrame:
    """
    Calculate statistical features for the segmented data
    :param data: segmented data
    :param column: column name
    :return: statistical features
    """

    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        print("Data not in right format")
        return None
    elif isinstance(data, pd.DataFrame):
        data = data.values

    mean = np.mean(data, axis=1).reshape((-1, 1))
    std = np.std(data, axis=1).reshape((-1, 1))
    min_ = np.amin(data, axis=1).reshape((-1, 1))
    max_ = np.amax(data, axis=1).reshape((-1, 1))
    range_ = abs(max_ - min_).reshape((-1, 1))
    q75, q25 = np.percentile(data, [75, 25], axis=1)
    iqr = q75 - q25
    iqr = iqr.reshape((-1, 1))
    kurtosis = stats.kurtosis(data, axis=1).reshape((-1, 1))
    skewness = stats.skew(data, axis=1).reshape((-1, 1))
    rms = np.sqrt(np.mean(data**2, axis=1)).reshape((-1, 1))

    statistical_features = np.hstack(
        (mean, std, min_, max_, range_, iqr, kurtosis, skewness, rms)
    )

    feature_names = [
        "mean",
        "std",
        "min_",
        "max_",
        "range_",
        "iqr",
        "kurtosis",
        "skewness",
        "rms",
    ]
    feature_names = [f"{column}_{feature}" for feature in feature_names]

    return pd.DataFrame(statistical_features, columns=feature_names)


def _calculate_frequency_features(
    data: Union[np.ndarray, pd.Series], column: str
) -> pd.DataFrame:
    """
    Calculate frequency features for the segmented data
    :param data: segmented data
    :param column: column name
    :return: frequency features
    """
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        print("Data not in right format")
        return None
    elif isinstance(data, pd.DataFrame):
        data = data.values

    fft = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[1])
    fft = fft[:, freq >= 0]
    freq = freq[freq >= 0]

    mean_freq = np.mean(freq * abs(fft), axis=1).reshape((-1, 1))
    std_freq = np.std(freq * abs(fft), axis=1).reshape((-1, 1))
    max_freq = np.amax(freq * abs(fft), axis=1).reshape((-1, 1))
    max_freq_mag = np.amax(abs(fft), axis=1).reshape((-1, 1))
    freq_mean = np.mean(abs(fft), axis=1).reshape((-1, 1))
    freq_std = np.std(abs(fft), axis=1).reshape((-1, 1))
    freq_skew = stats.skew(abs(fft), axis=1).reshape((-1, 1))
    freq_kurtosis = stats.kurtosis(abs(fft), axis=1).reshape((-1, 1))

    frequency_features = np.hstack(
        (
            mean_freq,
            std_freq,
            max_freq,
            max_freq_mag,
            freq_mean,
            freq_std,
            freq_skew,
            freq_kurtosis,
        )
    )

    feature_names = [
        "mean_freq",
        "std_freq",
        "max_freq",
        "max_freq_mag",
        "freq_mean",
        "freq_std",
        "freq_skew",
        "freq_kurtosis",
    ]
    feature_names = [f"{column}_{feature}" for feature in feature_names]

    return pd.DataFrame(frequency_features, columns=feature_names)


def calculate_features(
    data: pd.Series,
    columns: List[str],
    win_length: float,
    overlap: float,
    statistic_only: bool = False,
) -> pd.DataFrame:
    data_with_features_task = pd.DataFrame()
    for column in columns:
        segmented_sensor_data = _moving_window(data[column], win_length, overlap)
        segmented_sensor_data = pd.DataFrame(
            segmented_sensor_data, columns=[str(i) for i in range(win_length)]
        )
        segmented_sensor_data["index"] = np.arange(segmented_sensor_data.shape[0])
        column_order = ["index"] + [
            str(x) for x in range(segmented_sensor_data.shape[1] - 1)
        ]
        segmented_sensor_data = segmented_sensor_data[column_order]

        data_with_features_temp_stat = _calculate_statistical_features(
            segmented_sensor_data.iloc[:, 1:].values, column
        )
        if not statistic_only:
            data_with_features_temp_freq = _calculate_frequency_features(
                segmented_sensor_data.iloc[:, 1:].values, column
            )
        else:
            data_with_features_temp_freq = pd.DataFrame()
        
        data_with_features_task = pd.concat(
            [
                data_with_features_task,
                data_with_features_temp_stat,
                data_with_features_temp_freq,
            ],
            axis=1,
        )
    return data_with_features_task
