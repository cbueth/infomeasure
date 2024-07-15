"""Module for the Kozacenko-Leonenko entropy estimator."""

from numpy import inf
from numpy import sum as np_sum
from scipy.spatial import KDTree
from scipy.special import digamma

from ... import Config
from ...utils.types import LogBaseType
from ..base import EntropyEstimator, LogBaseMixin, RandomGeneratorMixin


class KozachenkoLeonenkoEstimator(LogBaseMixin, RandomGeneratorMixin, EntropyEstimator):
    r"""Kozachenko-Leonenko estimator for Shannon entropies.

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    k : int
        The number of nearest neighbors to consider.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    minkowski_p : float, :math:`1 \leq p \leq \infty`
        The power parameter for the Minkowski metric.
        Default is np.inf for maximum norm. Use 2 for Euclidean distance.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Methods
    -------
    calculate()
        Calculate the entropy.
    """

    def __init__(
        self,
        data,
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
        super().__init__(data, base=base)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        self.k = k
        self.noise_level = noise_level
        self.minkowski_p = minkowski_p

    def calculate(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """

        # Add small Gaussian noise to data to avoid issues with zero distances
        noise = self.rng.normal(0, self.noise_level, self.data.shape)
        data_noisy = self.data + noise

        # Build a KDTree for efficient nearest neighbor search with maximum norm
        tree = KDTree(data_noisy)  # KDTree uses 'Euclidean' metric by default

        # Find the k-th nearest neighbors for each point
        distances, _ = tree.query(data_noisy, self.k + 1, p=self.minkowski_p)

        # Exclude the zero distance to itself, which is the first distance
        distances = distances[:, self.k]

        # Constants for the entropy formula
        N = self.data.shape[0]
        d = self.data.shape[1]
        c_d = 1  # Volume of the d-dimensional unit ball for maximum norm

        # Compute the entropy estimator considering that the distances are
        # already doubled
        entropy = (
            -digamma(self.k)
            + digamma(N)
            + self._log_base(c_d)
            + (d / N) * np_sum(self._log_base(2 * distances))
        )

        return entropy
