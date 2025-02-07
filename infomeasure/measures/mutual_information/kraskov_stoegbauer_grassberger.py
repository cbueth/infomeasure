"""Module for the Kraskov-Stoegbauer-Grassberger (KSG) mutual information estimator."""

from abc import ABC
from numpy import column_stack, inf, array, ndarray
from numpy import newaxis
from scipy.spatial import KDTree
from scipy.special import digamma

from ..base import (
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
    EffectiveValueMixin,
)


class BaseKSGMIEstimator(ABC):
    r"""Base class for mutual information using the Kraskov-Stoegbauer-Grassberger (KSG)
    method.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the (conditional) mutual information.
    data_z : array-like, optional
        The conditional data used to estimate the conditional mutual information.
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
        Not compatible with the ``data_z`` parameter / conditional MI.
    normalize : bool, optional
        If True, normalize the data before analysis.
    base : "e"
        Only the natural logarithm is supported for the KSG estimator.
    """

    def __init__(
        self,
        data_x,
        data_y,
        data_z=None,
        *,  # all following parameters are keyword-only
        k: int = 4,
        noise_level=1e-10,
        minkowski_p=inf,
        offset: int = 0,
        normalize: bool = False,
        base="e",
    ):
        r"""Initialize the estimator with specific parameters.

        Parameters
        ----------
        data_x, data_y : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
        data_z : array-like, optional
            The conditional data used to estimate the conditional mutual information.
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
            Not compatible with the ``data_z`` parameter / conditional MI.
        base : "e"
            Only the natural logarithm is supported for the KSG estimator.
        """
        if base != "e":
            raise ValueError(
                "Only the natural logarithm is supported for the KSG estimator."
            )
        self.data_x = None
        self.data_y = None
        self.data_z = None
        if data_z is None:
            super().__init__(
                data_x, data_y, offset=offset, normalize=normalize, base=base
            )
        else:
            super().__init__(
                data_x, data_y, data_z, offset=offset, normalize=normalize, base=base
            )
            # Ensure self.data_z is a 2D array
            if self.data_z.ndim == 1:
                self.data_z = self.data_z[:, newaxis]
        # Ensure self.data_x and self.data_y are 2D arrays
        if self.data_x.ndim == 1:
            self.data_x = self.data_x[:, newaxis]
        if self.data_y.ndim == 1:
            self.data_y = self.data_y[:, newaxis]
        self.k = k
        self.noise_level = noise_level
        self.minkowski_p = minkowski_p


class KSGMIEstimator(
    BaseKSGMIEstimator, EffectiveValueMixin, MutualInformationEstimator
):
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
    normalize : bool, optional
        If True, normalize the data before analysis.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def _calculate(self) -> ndarray:
        """Calculate the mutual information of the data.

        Returns
        -------
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
        tree_joint = KDTree(data_joint)  # default leafsize is 10

        # Find the k-th nearest neighbor distance for each point in joint space using
        # the maximum norm
        distances, _ = tree_joint.query(data_joint, k=self.k + 1, p=self.minkowski_p)
        kth_distances = distances[:, -1]

        # Create KDTree objects for X and Y to count neighbors in marginal spaces using
        # the maximum norm
        tree_x = KDTree(self.data_x)
        tree_y = KDTree(self.data_y)

        # Count neighbors within k-th nearest neighbor distance in X and Y spaces using
        # the maximum norm
        count_x = [
            tree_x.query_ball_point(p, r=d, p=self.minkowski_p, return_length=True) - 1
            for p, d in zip(self.data_x, kth_distances)
        ]
        count_y = [
            tree_y.query_ball_point(p, r=d, p=self.minkowski_p, return_length=True) - 1
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

        return local_mi


class KSGCMIEstimator(BaseKSGMIEstimator, ConditionalMutualInformationEstimator):
    r"""Estimator for conditional mutual information using
    the Kraskov-Stoegbauer-Grassberger (KSG) method.

    Attributes
    ----------
    data_x, data_y, data_z : array-like
        The data used to estimate the conditional mutual information.
    k : int
        The number of nearest neighbors to consider.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    minkowski_p : float, :math:`1 \leq p \leq \infty`
        The power parameter for the Minkowski metric.
        Default is np.inf for maximum norm. Use 2 for Euclidean distance.
    normalize : bool, optional
        If True, normalize the data before analysis.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def _calculate(self) -> ndarray:
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        local_cmi : array
            Local conditional mutual information for each point.
        """
        # Copy the data to avoid modifying the original
        data_x_noisy = self.data_x.astype(float).copy()
        data_y_noisy = self.data_y.astype(float).copy()
        data_z_noisy = self.data_z.astype(float).copy()

        # Add Gaussian noise to the data if the flag is set
        if self.noise_level and self.noise_level != 0:
            data_x_noisy += self.rng.normal(0, self.noise_level, self.data_x.shape)
            data_y_noisy += self.rng.normal(0, self.noise_level, self.data_y.shape)
            data_z_noisy += self.rng.normal(0, self.noise_level, self.data_z.shape)

        # Stack the X, Y, and Z data to form joint observations
        data_joint = column_stack((data_x_noisy, data_y_noisy, data_z_noisy))

        # Create KDTree for efficient nearest neighbor search in joint space
        tree_joint = KDTree(data_joint)

        # Find k-th nearest neighbor distances in joint space
        distances, _ = tree_joint.query(data_joint, k=self.k + 1, p=self.minkowski_p)
        kth_distances = distances[:, -1]

        # Count points within k-th nearest neighbor distance in marginal spaces
        tree_xz = KDTree(column_stack((self.data_x, self.data_z)))
        tree_yz = KDTree(column_stack((self.data_y, self.data_z)))
        tree_z = KDTree(self.data_z)

        # Count neighbors within k-th nearest neighbor distance in X and Y spaces using
        # the maximum norm
        count_xz = [
            tree_xz.query_ball_point(p, r=d, p=self.minkowski_p, return_length=True) - 1
            for p, d in zip(column_stack((self.data_x, self.data_z)), kth_distances)
        ]
        count_yz = [
            tree_yz.query_ball_point(p, r=d, p=self.minkowski_p, return_length=True) - 1
            for p, d in zip(column_stack((self.data_y, self.data_z)), kth_distances)
        ]
        count_z = [
            tree_z.query_ball_point(p, r=d, p=self.minkowski_p, return_length=True) - 1
            for p, d in zip(self.data_z, kth_distances)
        ]

        # Compute local CMI for each data point
        return digamma(self.k) + array(
            [
                digamma(cz + 1) - digamma(cxz + 1) - digamma(cyz + 1)
                for cz, cxz, cyz in zip(count_z, count_xz, count_yz)
            ]
        )
