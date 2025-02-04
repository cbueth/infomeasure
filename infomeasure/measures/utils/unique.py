"""Utility functions for counting unique values."""

from numpy import ndarray, unique, zeros
from numpy import add as np_add


def histogram_unique_values(data: ndarray) -> ndarray:
    """Calculate the histogram of unique values in the data.

    Parameters
    ----------
    data : array-like
        The data to calculate the histogram for.

    Returns
    -------
    ndarray
        The histogram of unique values.
    """
    # Frequency Counting
    uniq, inverse = unique(data, return_inverse=True)
    histogram = zeros(len(uniq), int)  # NaNs are considered as a unique state
    np_add.at(histogram, inverse, 1)
    # Normalization
    return histogram / data.shape[0]
