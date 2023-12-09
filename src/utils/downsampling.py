import warnings
from typing import Union

import numpy as np
import pandas as pd
from scipy import signal


def _ffill(_array: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Function for expanding and array by forward filling the NaN values
    :param _array: contains the array to be expanded
    :return: the expanded array
    """

    if len(_array) == 0:
        raise ValueError("The sent array is empty.")
    array = pd.Series(_array).fillna(method="ffill")

    return np.array(array)


def downsample_signal(
    _data: Union[np.ndarray, pd.Series],
    fs: float,
    new_sample_rate: float,
    anti_aliasing_filter: str = "iir",
) -> np.ndarray:
    """
    Downsamples sensor data to a new sample rate using one of several methods
    with optional anti-aliasing filtering.
    :param _data: contains the data to be downsampled
    :param fs: containing the current sampling frequency
    :param new_sample_rate: containing the new sampling frequency
    :param anti_aliasing_filter: containing the anti-aliasing filter to use
    """

    if len(_data) == 0:
        raise ValueError("The sent data array is empty.")
    if new_sample_rate >= fs or new_sample_rate <= 0:
        raise ValueError(
            "New sampling frequency can't be higher or equal to the current"
            " frequency, and can't be lower or equal to 0."
        )
    if anti_aliasing_filter not in {None, "fir", "iir"}:
        raise ValueError(
            "Invalid anti_aliasing_filter value. Must be one of"
            " {None, 'fir', 'iir'}."
        )

    data = np.copy(_data)

    downsampling_factor = int(fs / new_sample_rate)
    if (fs / new_sample_rate) != downsampling_factor:
        warnings.warn(
            "The downsampling factor has a floating point and is rounded down."
            " This may cause the output to be in a slightly different sampling"
            " frequency."
        )

    if anti_aliasing_filter is None:
        data_downsampled = data[::downsampling_factor]
    else:
        data_downsampled = signal.decimate(
            data, downsampling_factor, ftype=anti_aliasing_filter
        )

    ffill_input = np.full(len(data), np.nan)

    ffill_input[
        [downsampling_factor * x for x in range(len(data_downsampled))]
    ] = data_downsampled

    ffill_downsampled_data = _ffill(ffill_input)

    return ffill_downsampled_data
