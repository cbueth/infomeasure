"""Symbolic / Permutation utility functions."""

from math import factorial

from numpy import (
    ndarray,
    argsort,
    apply_along_axis,
    zeros,
    sum as np_sum,
    array,
    uint64,
    iinfo,
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
    order : int, optional
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
        If the order is too large to convert to an uint64 (maximal 20).
    """
    n = len(perm)
    if n > 20:
        raise ValueError(
            "For orders larger than 20, the integer will be too large for uint64."
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
    series: ndarray, order: int, step_size: int = 1, to_int=False
) -> ndarray:
    """
    Convert a time series into a sequence of symbols (permutation patterns).

    Parameters
    ----------
    series : ndarray, shape (n,)
        A numpy array of data points.
    order : int
        The size of the permutation patterns.
    step_size : int
        The step size for the sliding windows. Takes every `step_size`-th element.
    to_int : bool, optional
        Whether to convert the permutation patterns to integers. Default is False.
        This

    Returns
    -------
    patterns : ndarray, shape (n - (order - 1) * step_size, order)
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
        If the order is less than 1.
    ValueError
        If the step_size is less than 1.
    """
    if order < 1:
        raise ValueError("The order must be a positive integer.")
    if step_size < 1:
        raise ValueError("The step_size must be a positive integer.")
    # Create a view of the series with the given order and step size
    shape = (series.size - (order - 1) * step_size, order)
    strides = (series.strides[0], series.strides[0] * step_size)
    # Extract subsequences
    subsequences = as_strided(series, shape=shape, strides=strides)
    # Get the permutation patterns
    patterns = apply_along_axis(argsort, 1, subsequences)

    # If Lehmer code is requested, convert the permutation to an integer
    if to_int:
        # Determine necessary dtype for the maximal number
        dtypes = ["uint8", "uint16", "uint32"]
        dtype = uint64
        for d in dtypes:  # try small to large
            if factorial(order) < iinfo(d).max:
                dtype = d
                break
        # Convert the permutation patterns to integers
        patterns = apply_along_axis(
            lambda x: permutation_to_integer(x, dtype=dtype), 1, patterns
        )

    return patterns
