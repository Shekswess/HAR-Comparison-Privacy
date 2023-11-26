from typing import Union

import numpy as np
import pandas as pd
import scipy.signal as sig


def calculate_mag(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Calculate magnitude of 3D accelerometer data
    :param data: accelerometer data
    :param columns: accelerometer columns
    :return: accelerometer data with magnitude column
    """
    mag = np.sqrt(np.sum(data[columns] ** 2, axis=1))
    mag_col_name = columns[0].split(".")[0] + ".Magnitude." \
        + columns[0].split(".")[2]
    data[mag_col_name] = mag
    return data


def _filter_design(
    order: int,
    fcritical: float,
    btype: str,
    fs: float,
    ftype: str = "butter",
    output: str = "ba",
):
    """
    Design a filter using scipy.signal
    :param order: filter order
    :param fcritical: critical frequency
    :param btype: filter type
    :param fs: sampling frequency
    :param ftype: filter type
    :param output: output type
    """
    if ftype.lower() == "fir":
        fcoefs = sig.firwin(order + 1, fcritical, fs=fs, pass_zero=btype)
    else:
        fcoefs = sig.iirfilter(
            order, fcritical, fs=fs, btype=btype, ftype=ftype, output=output
        )
    return fcoefs


def _filter_signal(
    data: Union[np.ndarray, pd.Series],
    fcoefs: Union[np.ndarray, tuple],
    zero_phase=True,
):
    """
    Filter a signal using scipy.signal
    :param data: signal to filter
    :param fcoefs: filter coefficients
    :param zero_phase: zero phase filter
    :return: filtered signal
    """
    if isinstance(data, pd.Series):
        data = data.values
    if isinstance(fcoefs, tuple):
        b, a = fcoefs
        filter_func = sig.filtfilt if zero_phase else sig.lfilter
        data_out = filter_func(b, a, data)
    elif isinstance(fcoefs, np.ndarray):
        if len(fcoefs.shape) == 1:
            b = fcoefs
            a = [1]
            filter_func = sig.filtfilt if zero_phase else sig.lfilter
            data_out = filter_func(b, a, data)
        elif len(fcoefs.shape) == 2 and fcoefs.shape[1] == 6:
            filter_func = sig.sosfiltfilt if zero_phase else sig.sosfilt
            data_out = filter_func(fcoefs, data)
        else:
            raise ValueError("Unknown filter coefficients type")
    else:
        raise ValueError("Unknown filter coefficients type")
    return data_out


def lowpass_filter(
    data: Union[np.ndarray, pd.Series],
    fs: float,
    fcritical: float,
    order: int = 3,
    ftype: str = "butter",
    output: str = "ba",
    zero_phase: bool = True,
):
    """
    Lowpass filter a signal using scipy.signal
    :param data: signal to filter
    :param fs: sampling frequency
    :param fcritical: critical frequency
    :param order: filter order
    :param ftype: filter type
    :param output: output type
    :param zero_phase: zero phase filter
    :return: filtered signal
    """
    fcoefs = _filter_design(order, fcritical, "lowpass", fs, ftype, output)
    data_out = _filter_signal(data, fcoefs, zero_phase)
    return data_out
