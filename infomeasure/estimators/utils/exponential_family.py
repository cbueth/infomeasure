"""Helper functions for exponential family distributions.

Rényi entropy and Tsallis entropy are special cases of the more general
family of exponential family distributions. This module provides helper
functions for these distributions.
"""

from numpy import asarray, pi, sum as np_sum, exp as np_exp
from scipy.spatial import KDTree
from scipy.special import gamma, digamma


def calculate_common_entropy_components(data, k, at=None):
    """Calculate common components for entropy estimators.

    Parameters
    ----------
    data : array-like
        The data used to estimate the entropy.
    k : int
        The number of nearest neighbors used in the estimation.
        Not including the data point itself.
    at : array-like, optional
        The parameter at which to evaluate the entropy components.

    Returns
    -------
    tuple
        Volume of the unit ball, k-th nearest neighbor distances,
        number of data points, and dimensionality of the data.

    Raises
    ------
    ValueError
        If the parameter ``k`` is selected too large.
    ValueError
        If both ``data`` and ``at`` have different dimensions.
    """
    N, m = data.shape

    if at is None:
        at = data
        k += 1  # Exclude the data point itself in the nearest neighbors calculation
    elif at.shape[1] != m:
        raise ValueError(
            "The data and parameter at which to evaluate "
            "the entropy components must have the same dimensionality."
        )

    if k > N:
        raise ValueError(
            "The number of nearest neighbors must be smaller "
            "than the number of data points."
        )

    # Volume of the unit ball in m-dimensional space
    V_m = pi ** (m / 2) / gamma(m / 2 + 1)

    # Build k-d tree for nearest neighbor search
    tree = KDTree(data)

    # Get the k-th nearest neighbor distances.
    # KDTree.query returns a 1-D array when k == 1.
    rho_k = asarray(tree.query(at, k=k)[0])
    if rho_k.ndim == 1:
        rho_k = rho_k[:, None]
    rho_k = rho_k[:, k - 1]

    return V_m, rho_k, N, m
