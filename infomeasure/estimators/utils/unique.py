""""""

from typing import Any

from numpy import unique, ndarray, dtype, signedinteger
from numpy._typing import _32Bit, _64Bit
from numpy.lib._arraysetops_impl import _SCT


def unique_vals(
    data: ndarray,
) -> tuple[
    ndarray[tuple[int, ...], dtype[_SCT]],
    ndarray[tuple[int, ...], dtype[signedinteger[_32Bit | _64Bit]]],
    dict[Any, Any],
]:
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
    probability = counts / len(data[0])
    dist = dict(zip(uniq, probability))
    return uniq, counts, dist
