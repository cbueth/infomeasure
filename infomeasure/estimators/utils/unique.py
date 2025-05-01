"""Module for handling symbolic/discrete values."""

from numpy import unique, ndarray


def unique_vals(data: ndarray) -> tuple[ndarray, ndarray, dict]:
    """
    Get unique values and their counts and probability distribution.

    Parameters
    ----------
    data : ndarray
        Data to get unique values from.

    Returns
    -------
    tuple
        Unique values, their counts, and probability distribution.
    """
    uniq, counts = unique(data, return_counts=True)
    probability = counts / len(data)
    dist = dict(zip(uniq, probability))
    return uniq, counts, dist
