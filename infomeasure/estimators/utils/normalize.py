"""Normalize a time series."""

from numpy import max as np_max
from numpy import min as np_min
from numpy import ndarray


def normalize_data_0_1(data: ndarray) -> ndarray:
    """
    Normalize the given data to scale each dimension to the range [0, 1].

    Parameters
    ----------
    data : 2D array-like
        List of data points where each row is a data point in d-dimensional space.

    Returns
    -------
    2D array
        Data scaled to the range [0, 1].

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4], [5, 6]])
    >>> normalize_data_0_1(data)
    array([[0. , 0. ],
           [0.5, 0.5],
           [1. , 1. ]])
    """
    min_vals = np_min(data, axis=0)
    max_vals = np_max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)
