"""Module for the Kozachenko-Leonenko entropy estimator."""

from numpy import column_stack, sum as np_sum, nan, isnan
from numpy import inf, log, issubdtype, integer
from scipy.spatial import KDTree
from scipy.special import digamma

from ..base import EntropyEstimator, RandomGeneratorMixin
from ..utils.array import assure_2d_data
from ..utils.unit_ball_volume import unit_ball_volume
from ... import Config
from ...utils.types import LogBaseType


class KozachenkoLeonenkoEntropyEstimator(RandomGeneratorMixin, EntropyEstimator):
    r"""Kozachenko-Leonenko estimator for Shannon entropies.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    k : int
        The number of nearest neighbors to consider.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    minkowski_p : float, :math:`1 \leq p \leq \infty`
        The power parameter for the Minkowski metric.
        Default is np.inf for maximum norm. Use 2 for Euclidean distance.

    Raises
    ------
    ValueError
        If the number of nearest neighbors is not a positive integer
    ValueError
        If the noise level is negative
    ValueError
        If the Minkowski power parameter is invalid

    Notes
    -----
    Changing the number of nearest neighbors ``k`` can change the outcome,
    but the default value of :math:`k=4` is recommended by :cite:p:`miKSG2004`.
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
