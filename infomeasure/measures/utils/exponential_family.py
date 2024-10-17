"""Helper functions for exponential family distributions.

Rényi entropy and Tsallis entropy are special cases of the more general
family of exponential family distributions. This module provides helper
functions for these distributions.
"""

from numpy import pi, mean as np_mean, exp as np_exp
from scipy.special import gamma, digamma
from scipy.spatial import KDTree


def calculate_common_entropy_components(data, k):
    """Calculate common components for entropy estimators.

    Parameters
    ----------
    data : array-like
        The data used to estimate the entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    q : float | int
        The Rényi or Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.

    Returns
    -------
    tuple
        Volume of the unit ball, k-th nearest neighbor distances,
        number of data points, and dimensionality of the data.
    """
    N, m = data.shape

    # Volume of the unit ball in m-dimensional space
    V_m = pi ** (m / 2) / gamma(m / 2 + 1)

    # Build k-d tree for nearest neighbor search
    tree = KDTree(data)

    # Get the k-th nearest neighbor distances
    rho_k = tree.query(data, k=k + 1)[0][
        :, k
    ]  # k+1 because the point itself is included

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
        Number of data points.
    m : int
        Dimensionality of the data.

    Returns
    -------
    float
        The :math:`I_q` of the exponential family distribution
    """
    C_k = (gamma(k) / gamma(k + 1 - q)) ** (1 / (1 - q))
    zeta_N_i_k = (N - 1) * C_k * V_m * rho_k**m
    return np_mean(zeta_N_i_k ** (1 - q))


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
        Number of data points.
    m : int
        Dimensionality of the data.
    log_base_func : callable
        The logarithm function to use for the calculation with the chosen base.

    Returns
    -------
    float
        The :math:`I_1` of the exponential family distribution
    """
    zeta_N_i_k = (N - 1) * np_exp(-digamma(k)) * V_m * rho_k**m
    return np_mean(log_base_func(zeta_N_i_k))
