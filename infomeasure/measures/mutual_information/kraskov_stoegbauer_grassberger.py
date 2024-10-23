"""Module for the Kraskov-Stoegbauer-Grassberger (KSG) mutual information estimator."""

from numpy import column_stack, inf, array
from numpy import mean as np_mean
from numpy import newaxis
from scipy.spatial import KDTree
from scipy.special import digamma

from ... import Config
from ...utils.types import LogBaseType
from ..base import (
    MutualInformationEstimator,
    EffectiveValueMixin,
)


class KSGMIEstimator(EffectiveValueMixin, MutualInformationEstimator):
    r"""Estimator for mutual information using the Kraskov-Stoegbauer-Grassberger (KSG)
    method.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    k : int
        The number of nearest neighbors to consider.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    minkowski_p : float, :math:`1 \leq p \leq \infty`
        The power parameter for the Minkowski metric.
        Default is np.inf for maximum norm. Use 2 for Euclidean distance.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    normalize
        If True, normalize the data before analysis.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def __init__(
        self,
        data_x,
        data_y,
        k: int = 4,
        noise_level=1e-10,
        minkowski_p=inf,
        offset: int = 0,
        normalize: bool = False,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the estimator with specific parameters.

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
        normalize
            If True, normalize the data before analysis.
        offset : int, optional
            Number of positions to shift the data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
        """
        super().__init__(data_x, data_y, offset=offset, normalize=normalize, base=base)
        if self.data_x.ndim == 1:
            self.data_x = self.data_x.reshape(-1, 1)
        if self.data_y.ndim == 1:
            self.data_y = self.data_y.reshape(-1, 1)
        self.k = k
        self.noise_level = noise_level
        self.minkowski_p = minkowski_p

        # Ensure the data is 2D for KDTree
        if self.data_x.ndim == 1:
            self.data_x = self.data_x[:, newaxis]
        if self.data_y.ndim == 1:
            self.data_y = self.data_y[:, newaxis]

    def _calculate(self) -> tuple:
        """Calculate the mutual information of the data.

        Returns
        -------
        mi : float
            Estimated mutual information between the two datasets.
        local_mi : array
            Local mutual information for each point.
        """
        # Copy the data to avoid modifying the original
        data_x_noisy = self.data_x.astype(float).copy()
        data_y_noisy = self.data_y.astype(float).copy()

        # Add Gaussian noise to the data if the flag is set
        if self.noise_level and self.noise_level != 0:
            data_x_noisy += self.rng.normal(0, self.noise_level, self.data_x.shape)
            data_y_noisy += self.rng.normal(0, self.noise_level, self.data_y.shape)

        # Stack the X and Y data to form joint observations
        data_joint = column_stack((data_x_noisy, data_y_noisy))

        # Create a KDTree for joint data to find nearest neighbors using the maximum
        # norm
        tree_joint = KDTree(data_joint, leafsize=10)  # default leafsize is 10

        # Find the k-th nearest neighbor distance for each point in joint space using
        # the maximum norm
        distances, _ = tree_joint.query(data_joint, k=self.k + 1, p=self.minkowski_p)
        kth_distances = distances[:, -1]

        # Create KDTree objects for X and Y to count neighbors in marginal spaces using
        # the maximum norm
        tree_x = KDTree(self.data_x, leafsize=10)
        tree_y = KDTree(self.data_y, leafsize=10)

        # Count neighbors within k-th nearest neighbor distance in X and Y spaces using
        # the maximum norm
        count_x = [
            len(tree_x.query_ball_point(p, r=d, p=self.minkowski_p)) - 1
            for p, d in zip(self.data_x, kth_distances)
        ]
        count_y = [
            len(tree_y.query_ball_point(p, r=d, p=self.minkowski_p)) - 1
            for p, d in zip(self.data_y, kth_distances)
        ]

        # Compute mutual information using the KSG estimator formula
        N = len(self.data_x)
        # Compute local mutual information for each point
        local_mi = array(
            [
                digamma(self.k) - digamma(nx + 1) - digamma(ny + 1) + digamma(N)
                for nx, ny in zip(count_x, count_y)
            ]
        )

        # Compute aggregated mutual information
        mi = np_mean(local_mi)

        return mi, local_mi
