"""Utility functions for counting unique values."""

from typing import Tuple, Dict

from numpy import ndarray, unique, zeros
from numpy import add as np_add


def histogram_unique_values(data: ndarray) -> Tuple[ndarray[float], Dict[int, float]]:
    """Calculate the histogram of unique values in the data.

    Parameters
    ----------
    data : array-like
        The data to calculate the histogram for.

    Returns
    -------
    ndarray
        The histogram of unique values.
    dict
        The distribution of unique values.
        dict(unique_value: probability)
    """
    # Frequency Counting
    uniq, inverse = unique(data, return_inverse=True)
    histogram = zeros(len(uniq), int)  # NaNs are considered as a unique state
    np_add.at(histogram, inverse, 1)
    # Normalization
    histogram = histogram / data.shape[0]
    dist_dict = dict(zip(uniq, histogram))
    return histogram, dist_dict
