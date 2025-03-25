"""Ordinal / Permutation utility functions."""

from math import factorial
from typing import Generator

from numpy import (
    ndarray,
    argsort,
    apply_along_axis,
    zeros,
    sum as np_sum,
    array,
    uint64,
    iinfo,
    unique,
    column_stack,
)
from numpy.lib.stride_tricks import as_strided


def permutation_to_integer(perm: ndarray, dtype: type = uint64) -> int:
    """
    Convert a permutation pattern to a unique integer.
    The Lehmer code is used to convert the permutation to an integer.

    Parameters
    ----------
    perm : ndarray
        A permutation pattern.
    embedding_dim : int, optional
        The size of the permutation pattern. Default is
        None, which uses the length of the permutation.
        Using this, the maximal number will be set and the smnalles possible
        dtype will be used.

    Returns
    -------
    int : int, uint8, uint16, uint32, uint64
        A unique integer representing the permutation pattern.

    Examples
    --------
    >>> permutation_to_integer(array([0, 1]))
    0
    >>> permutation_to_integer(array([1, 0]))
    1
    >>> permutation_to_integer(array([0, 1, 2]))
    0
    >>> permutation_to_integer(array([2, 1, 0]))
    5

    Notes
    -----
    This approach has at least been known since
    1888 :cite:p:`laisantNumerationFactorielleApplication1888`.
    It is named after Derrick Henry Lehmer :cite:p:`Lehmer1960TeachingCT`.

    Raises
    ------
    ValueError
        If the embedding_dim is too large to convert to an uint64 (maximal 20).
    """
    n = len(perm)
    if n > 20:
        raise ValueError(
            "For embedding dimensions larger than 20, "
            "the integer will be too large for uint64."
        )
    factoradic = zeros(n, dtype=dtype)
    for i in range(n):
        factoradic[i] = np_sum(perm[i] > perm[i + 1 :], dtype=dtype)
    integer = np_sum(
        factoradic * array([factorial(n - 1 - i) for i in range(n)]),
        dtype=dtype,
    )
    return integer


def symbolize_series(
    series: ndarray, embedding_dim: int, step_size: int = 1, to_int=False, stable=False
) -> ndarray:
    """
    Convert a time series into a sequence of symbols (permutation patterns).

    Parameters
    ----------
    series : ndarray, shape (n,)
        A numpy array of data points.
    embedding_dim : int
        The size of the permutation patterns.
    step_size : int
        The step size for the sliding windows. Takes every `step_size`-th element.
    to_int : bool, optional
        Whether to convert the permutation patterns to integers. Default is False.
        This
    stable : bool, optional
        If True, when sorting the data, the embedding_dim of equal elements is preserved.
        This can be useful for reproducibility and testing, but might be slower.

    Returns
    -------
    patterns : ndarray, shape (n - (embedding_dim - 1) * step_size, embedding_dim)
        A list of tuples representing the symbolized series.

    Examples
    --------
    >>> series = np.array([1, 2, 3, 2, 1])
    >>> symbolize_series(series, 2, 1)
    array([[0, 1],
           [0, 1],
           [1, 0],
           [1, 0]])

    Raises
    ------
    ValueError
        If the embedding_dim is less than 1.
    ValueError
        If the step_size is less than 1.
    """
    if embedding_dim < 1:
        raise ValueError("The embedding_dim must be a positive integer.")
    if step_size < 1:
        raise ValueError("The step_size must be a positive integer.")
    # Create a view of the series with the given embedding_dim and step size
    shape = (series.size - (embedding_dim - 1) * step_size, embedding_dim)
    strides = (series.strides[0], series.strides[0] * step_size)
    # Extract subsequences
    subsequences = as_strided(series, shape=shape, strides=strides)
    # Get the permutation patterns
    patterns = apply_along_axis(argsort, 1, subsequences, stable=stable)

    # If Lehmer code is requested, convert the permutation to an integer
    if to_int:
        # Determine necessary dtype for the maximal number
        dtypes = ["uint8", "uint16", "uint32"]
        dtype = uint64
        for d in dtypes:  # try small to large
            if factorial(embedding_dim) < iinfo(d).max:
                dtype = d
                break
        # Convert the permutation patterns to integers
        patterns = apply_along_axis(
            lambda x: permutation_to_integer(x, dtype=dtype), 1, patterns
        )

    return patterns


def reduce_joint_space(data: ndarray | tuple[ndarray] | Generator) -> ndarray:
    """Reduce the data to the joint space.

    Reduce features: Assigns each unique feature vector to a unique integer.
    This is equivalent to the unique indices of the unique rows.
    If `data` is 1D, returns the data as is.

    Parameters
    ----------
    data : ndarray, shape (n_samples,) or (n_samples, n_features), or tuple of arrays
        The data to reduce.

    Returns
    -------
    ndarray, shape (n_samples,)
        The data in the joint space.

    Examples
    --------
    >>> from numpy import array
    >>> data = array([[1, 2], [2, 3], [1, 2], [2, 3], [3, 4]])
    >>> data.shape
    (5, 2)
    >>> reduce_joint_space(data)
    array([0, 1, 0, 1, 2])
    >>> reduce_joint_space(array([4, 5, 4, 5, 6]))
    array([4, 5, 4, 5, 6])
    >>> reduce_joint_space(array([[True, True, False], [False, True, False]]))
    array([0, 1])
    >>> reduce_joint_space(array([[3, 3], [2, 2], [1, 1]]))
    array([2, 1, 0])
    >>> reduce_joint_space((array([3, 2, 1]), array([3, 2, 1])))
    array([2, 1, 0])

    Raises
    ------
    ValueError
        If the data array is not 1D or 2D.
    ValueError
        If the data is a list.

    Notes
    -----
    The order of the unique values is not guaranteed, only the unique indices.
    This is because numpy sorts the values internally.
    """
    if isinstance(data, list):
        raise ValueError("Data array must be a numpy array or tuple of arrays.")
    if isinstance(data, Generator):
        data = tuple(data)
    if isinstance(data, tuple):
        data = column_stack(data)
    if data.ndim == 2:
        # Reduce the data to the joint space using the numpy unique function
        return unique(data, axis=0, return_inverse=True)[1]  # use inverse indices
    elif data.ndim == 1:
        return data
    raise ValueError("Data array must be 1D or 2D.")
