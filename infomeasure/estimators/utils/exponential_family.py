"""Helper functions for exponential family distributions.

Rényi entropy and Tsallis entropy are special cases of the more general
family of exponential family distributions. This module provides helper
functions for these distributions.
"""

from numpy import pi, sum as np_sum, exp as np_exp
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

    # Get the k-th nearest neighbor distances
    rho_k = tree.query(at, k=k)[0][:, k - 1]  # k+1 because the point itself is included

    return V_m, rho_k, N, m


def exponential_family_iq(k, q, V_m, rho_k, N, m):
    r"""Calculate the :math:`I_q` of the exponential family distribution.

    Parameters
    ----------
    k : int
        The number of nearest neighbors used in the estimation.
    q : float | int
        The Rényi or Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.
        Should not be 1.
    V_m : float
        Volume of the unit ball in m-dimensional space.
    rho_k : array-like
        The k-th nearest neighbor distances.
    N : int
        Number of data points considered for the distances
        (Subtract 1 if own point not considered).
    m : int
        Dimensionality of the data.

    Returns
    -------
    float
        The :math:`I_q` of the exponential family distribution
    """
    if q == k + 1:  # In this case, C_k is complex infinite and (C_k)^-q = 0 for real q.
        return 0.0
    C_k = (gamma(k) / gamma(k + 1 - q)) ** (1 / (1 - q))
    return (
        (N * C_k * V_m) ** (1 - q)
        * np_sum((rho_k[rho_k > 0] ** m) ** (1 - q))
        / len(rho_k)
    )


def exponential_family_i1(k, V_m, rho_k, N, m, log_base_func):
    r"""Calculate the :math:`I_1` of the exponential family distribution.

    When :math:`q = 1`, the exponential family distribution reduces to the
    Shannon entropy.

    Parameters
    ----------
    k : int
        The number of nearest neighbors used in the estimation.
    V_m : float
        Volume of the unit ball in m-dimensional space.
    rho_k : array-like
        The k-th nearest neighbor distances.
    N : int
        Number of data points considered for the distances
        (Subtract 1 if own point not considered).
    m : int
        Dimensionality of the data.
    log_base_func : callable
        The logarithm function to use for the calculation with the chosen base.

    Returns
    -------
    float
        The :math:`I_1` of the exponential family distribution
    """
    zeta_N_i_k = N * np_exp(-digamma(k)) * V_m * rho_k**m
    return np_sum(log_base_func(zeta_N_i_k[zeta_N_i_k > 0])) / len(zeta_N_i_k)
    # return log_base_func(  # Analytically correct and efficient but inexact
    #     prod(rho_k[rho_k > 0] ** m)
    #     * (N * np_exp(-digamma(k)) * V_m) ** len(rho_k[rho_k > 0])
    # ) / len(rho_k)
