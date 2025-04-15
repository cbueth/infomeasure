"""Array helpers for the base estimator."""

from typing import Generator

from numpy import ndarray, newaxis, asarray


def assure_2d_data(data) -> ndarray | tuple[ndarray, ...]:
    """Assure the data is 2D.

    This function checks adds a new axis to the data if it is 1D.
    For tuples and generators, each element is checked and reshaped if necessary.

    Parameters
    ----------
    data : array-like or tuple/Generator of array-like
        The data to check.

    Returns
    -------
    array-like or tuple of array-like
        The reshaped data.

    Raises
    ------
    ValueError
        If the data is not supported for 2D conversion.
    """
    if isinstance(data, list):
        data = asarray(data)
    if isinstance(data, ndarray):
        if data.ndim == 1:
            return data[:, newaxis]
        return data
    if isinstance(data, Generator):
        data = tuple(data)
    if isinstance(data, tuple):
        data = (asarray(val) for val in data)
        return tuple(val[:, newaxis] if val.ndim == 1 else val for val in data)
    raise ValueError(f"Data type {type(data)} is not supported for 2D conversion.")
