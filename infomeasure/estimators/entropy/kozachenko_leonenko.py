"""Module for the Kozachenko-Leonenko entropy estimator."""

from numpy import column_stack, sum as np_sum, nan, isnan
from numpy import inf, log, issubdtype, integer
from scipy.spatial import KDTree
from scipy.special import digamma

from ..base import EntropyEstimator
from ..mixins import RandomGeneratorMixin
from ..utils.array import assure_2d_data
from ..utils.unit_ball_volume import unit_ball_volume
from ... import Config
from ...utils.types import LogBaseType


class KozachenkoLeonenkoEntropyEstimator(RandomGeneratorMixin, EntropyEstimator):
    r"""Kozachenko-Leonenko entropy estimator for continuous data.

    The Kozachenko-Leonenko estimator computes the Shannon entropy of continuous
    data using nearest neighbor distances. The estimator is based on the method
    from :cite:p:`kozachenko1987sample` and follows the implementation approach
    described in :cite:p:`miKSG2004`.

    .. math::

        \hat{H}_{KL} = -\psi(k) + \psi(N) + \log(c_d) + \frac{d}{N} \sum_{i=1}^{N} \log(2\rho_{k,i})

    where :math:`\psi` is the digamma function, :math:`k` is the number of nearest
    neighbors, :math:`N` is the number of data points, :math:`d` is the dimensionality,
    :math:`c_d` is the volume of the :math:`d`-dimensional unit ball for the chosen
    Minkowski norm, and :math:`\rho_{k,i}` is the distance to the :math:`k`-th nearest
    neighbor of point :math:`i`.

    This estimator is particularly suitable for continuous multivariate data and
    provides asymptotically unbiased estimates of differential entropy. The method
    works by exploiting the relationship between nearest neighbor distances and
    local density, making it effective for high-dimensional data where traditional
    histogram-based methods fail.

    Parameters
    ----------
    *data : array-like
        The continuous data used to estimate the entropy. For multivariate data,
        each variable should be a column.
    k : int, default=4
        The number of nearest neighbors to consider. Higher values provide more
        stable estimates but may introduce bias. The default value of 4 is
        recommended by :cite:p:`miKSG2004`.
    noise_level : float, default=1e-10
        The standard deviation of Gaussian noise added to the data to avoid
        issues with zero distances between identical points. Set to 0 to disable
        noise addition.
    minkowski_p : float, default=inf
        The power parameter for the Minkowski metric used in distance calculations.
        Common values are 2 (Euclidean distance) and inf (maximum norm/Chebyshev
        distance). Must satisfy :math:`1 \leq p \leq \infty`.
    base : LogBaseType, default=Config.get("base")
        The logarithm base for entropy calculation. Can be 2, 10, "e", or any
        positive number.

    Attributes
    ----------
    *data : tuple[array-like]
        The processed data used to estimate the entropy, converted to 2D arrays.
    k : int
        The number of nearest neighbors to consider.
    noise_level : float
        The standard deviation of the Gaussian noise added to the data.
    minkowski_p : float
        The power parameter for the Minkowski metric.

    Raises
    ------
    ValueError
        If the number of nearest neighbors is not a positive integer.
    ValueError
        If the noise level is negative.
    ValueError
        If the Minkowski power parameter is invalid (not in range [1, ∞]).

    Notes
    -----
    The choice of the number of nearest neighbors :math:`k` affects the bias-variance
    tradeoff of the estimator. Smaller values of :math:`k` reduce bias but increase
    variance, while larger values have the opposite effect. The default value of
    :math:`k=4` provides a good balance for most applications.

    The noise addition helps handle datasets with repeated values or points that
    are exactly identical, which would otherwise result in zero distances and
    numerical issues. The noise level should be small enough not to significantly
    alter the underlying distribution.

    For high-dimensional data, the curse of dimensionality may affect the estimator's
    performance, as nearest neighbor distances become less informative. In such cases,
    dimensionality reduction or alternative entropy estimation methods may be preferable.

    Examples
    --------
    >>> import numpy as np
    >>> import infomeasure as im
    >>>
    >>> # Generate 2D Gaussian data
    >>> np.random.seed(176250)
    >>> data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 1000)
    >>>
    >>> # Estimate entropy
    >>> estimator = im.estimator(data, measure="h", approach="kl", k=4)
    >>> entropy_value = estimator.result()
    >>> print(f"Estimated entropy: {entropy_value:.3f}")
    Estimated entropy: 2.678
    >>> print(f"Local values: {estimator.local_vals()}")
    Local values: [ 3.15330798  2.02688591  2.52250064  2.95236651  3.58801879  1.42033673
        ...
        2.91254223  1.92823136  3.63647704  2.05589055]
    >>> # Use different distance metric
    >>> estimator_euclidean = KozachenkoLeonenkoEntropyEstimator(data, k=4, minkowski_p=2)
    >>> entropy_euclidean = estimator_euclidean.entropy()
    np.float64(2.6772465397252208)
    """

    def __init__(
        self,
        *data,
        k: int = 4,
        noise_level=1e-10,
        minkowski_p=inf,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the Kozachenko-Leonenko estimator.

        Parameters
        ----------
        k : int
            The number of nearest neighbors to consider.
        noise_level : float
            The standard deviation of the Gaussian noise to add to the data to avoid
            issues with zero distances.
        minkowski_p : float, :math:`1 \leq p \leq \infty`
            The power parameter for the Minkowski metric.
            Default is np.inf for maximum norm. Use 2 for Euclidean distance.
        """
        if not issubdtype(type(k), integer) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors (k) must be a positive "
                f"integer, but got {k}."
            )
        if noise_level < 0:
            raise ValueError(
                f"The noise level must be non-negative, but got {noise_level}."
            )
        if not (1 <= minkowski_p <= inf):
            raise ValueError(
                "The Minkowski power parameter must be positive, "
                f"but got {minkowski_p}."
            )
        super().__init__(*data, base=base)
        self.data = tuple(assure_2d_data(var) for var in self.data)
        self.k = k
        self.noise_level = noise_level
        self.minkowski_p = minkowski_p

    def _simple_entropy(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        # Copy the data to avoid modifying the original
        data_noisy = self.data[0].astype(float).copy()
        # Add small Gaussian noise to data to avoid issues with zero distances
        if self.noise_level and self.noise_level != 0:
            data_noisy += self.rng.normal(0, self.noise_level, self.data[0].shape)

        # Build a KDTree for efficient nearest neighbor search with maximum norm
        tree = KDTree(data_noisy)

        # Find the k-th nearest neighbors for each point
        distances, _ = tree.query(data_noisy, self.k + 1, p=self.minkowski_p)
        # Only keep the k-th nearest neighbor distance
        distances = distances[:, -1]

        # Constants for the entropy formula
        N = self.data[0].shape[0]
        d = self.data[0].shape[1]
        # Volume of the d-dimensional unit ball for maximum norm
        c_d = unit_ball_volume(d, r=1 / 2, p=self.minkowski_p)

        distances[distances == 0] = nan
        # Compute the local entropies
        local_h = -digamma(self.k) + digamma(N) + log(c_d) + d * log(2 * distances)
        local_h[isnan(local_h)] = 0.0
        # return in desired base
        return local_h / log(self.base) if self.base != "e" else local_h

    def _joint_entropy(self):
        """Calculate the joint entropy of the data.

        This is done by joining the variables into one space
        and calculating the entropy.

        Returns
        -------
        float
            The calculated joint entropy.
        """
        self.data = (column_stack(self.data[0]),)
        return self._simple_entropy()

    def _cross_entropy(self) -> float:
        """Calculate the cross-entropy between two distributions.

        Returns
        -------
        float
            The calculated cross-entropy.
        """

        # Copy the data to avoid modifying the original
        data_noisy_p = self.data[0].astype(float).copy()
        data_noisy_q = self.data[1].astype(float).copy()
        # Add small Gaussian noise to data to avoid issues with zero distances
        if self.noise_level and self.noise_level != 0:
            data_noisy_p += self.rng.normal(0, self.noise_level, self.data[0].shape)
            data_noisy_q += self.rng.normal(0, self.noise_level, self.data[1].shape)

        # Build a KDTree for efficient nearest neighbor search with maximum norm
        tree = KDTree(data_noisy_q)

        # Find the k-th nearest neighbors for each point
        distances, _ = tree.query(data_noisy_p, self.k, p=self.minkowski_p)
        # Only keep the k-th nearest neighbor distance
        distances = distances[:, -1]

        # Constants for the entropy formula
        M = self.data[1].shape[0]
        d = self.data[1].shape[1]
        # Volume of the d-dimensional unit ball for maximum norm
        c_d = unit_ball_volume(d, r=1 / 2, p=self.minkowski_p)

        # Compute the cross-entropy
        hx = (
            -digamma(self.k)
            + digamma(M)
            + log(c_d)
            + d * np_sum(log(2 * distances[distances > 0])) / M
        )
        # return in desired base
        return hx / log(self.base) if self.base != "e" else hx
