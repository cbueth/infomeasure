"""Module for the Kraskov-Stoegbauer-Grassberger (KSG) mutual information estimator."""

from abc import ABC

from numpy import column_stack, inf, array, ndarray, log
from scipy.spatial import KDTree
from scipy.special import digamma

from ..base import (
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
    PValueMixin,
)
from ..utils.array import assure_2d_data
from ... import Config
from ...utils.types import LogBaseType


class BaseKSGMIEstimator(ABC):
    r"""Base class for mutual information using the Kraskov-Stoegbauer-Grassberger (KSG)
    method.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
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
        Not compatible with the ``cond`` parameter / conditional MI.
    normalize : bool, optional
        If True, normalize the data before analysis.
    """

    def __init__(
        self,
        *data,
        cond=None,
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
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
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
            Not compatible with the ``cond`` parameter / conditional MI.
        """
        self.data: tuple[ndarray] = None
        self.cond = None
        if cond is None:
            super().__init__(*data, offset=offset, normalize=normalize, base=base)
        else:
            super().__init__(
                *data, cond=cond, offset=offset, normalize=normalize, base=base
            )
            # Ensure self.cond is a 2D array
            self.cond = assure_2d_data(self.cond)
        self.k = k
        self.noise_level = noise_level
        self.minkowski_p = minkowski_p
        # Ensure self.data_x and self.data_y are 2D arrays
        self.data = tuple(assure_2d_data(var) for var in self.data)


class KSGMIEstimator(BaseKSGMIEstimator, PValueMixin, MutualInformationEstimator):
    r"""Estimator for mutual information using the Kraskov-Stoegbauer-Grassberger (KSG)
    method.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
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

    Notes
    -----
    Changing the number of nearest neighbors ``k`` can change the outcome,
    but the default value of :math:`k=4` is recommended by :cite:p:`miKSG2004`.
    """

    def _calculate(self) -> ndarray:
        """Calculate the mutual information of the data.

        Returns
        -------
        local_mi : array
            Local mutual information for each point.
        """
        # Copy the data to avoid modifying the original
        data = [var.astype(float).copy() for var in self.data]

        # Add Gaussian noise to the data if the flag is set
        if self.noise_level and self.noise_level != 0:
            for i in range(len(data)):
                for j in range(len(data[i])):
                    data[i][j] += self.rng.normal(0, self.noise_level, data[i][j].shape)

        # Stack the X and Y data to form joint observations
        data_joint = column_stack(data)

        # Create a KDTree for joint data to find nearest neighbors using the maximum
        # norm
        tree_joint = KDTree(data_joint)  # default leafsize is 10

        # Find the k-th nearest neighbor distance for each point in joint space using
        # the maximum norm
        distances, _ = tree_joint.query(data_joint, k=self.k + 1, p=self.minkowski_p)
        kth_distances = distances[:, -1]

        # Create KDTree objects for X and Y to count neighbors in marginal spaces using
        # the maximum norm
        trees_marginal = [KDTree(var) for var in data]

        # Count neighbors within k-th nearest neighbor distance in X and Y spaces using
        # the maximum norm
        counts_marginal = [
            [
                tree.query_ball_point(p, r=d, p=self.minkowski_p, return_length=True)
                - 1
                for p, d in zip(var, kth_distances)
            ]
            for tree, var in zip(trees_marginal, data)
        ]

        # Compute mutual information using the KSG estimator formula
        N = len(data[0])
        m = len(data)  # number of variables
        # Compute local mutual information for each point
        local_mi = array(
            [
                digamma(self.k)
                - sum(digamma(n + 1) for n in counts)
                + (m - 1) * digamma(N)
                for counts in zip(*counts_marginal)
            ]
        )

        return local_mi / log(self.base) if self.base != "e" else local_mi


class KSGCMIEstimator(BaseKSGMIEstimator, ConditionalMutualInformationEstimator):
    r"""Estimator for conditional mutual information using
    the Kraskov-Stoegbauer-Grassberger (KSG) method.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
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

    Notes
    -----
    Changing the number of nearest neighbors ``k`` can change the outcome,
    but the default value of :math:`k=4` is recommended by :cite:p:`miKSG2004`.
    """

    def _calculate(self) -> ndarray:
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        local_cmi : array
            Local conditional mutual information for each point.
        """
        # Copy the data to avoid modifying the original
        data = [var.astype(float).copy() for var in self.data]
        cond = self.cond.astype(float).copy()

        # Add Gaussian noise to the data if the flag is set
        if self.noise_level and self.noise_level != 0:
            for j in range(len(data[0])):
                for i in range(len(data)):
                    data[i][j] += self.rng.normal(0, self.noise_level, data[i][j].shape)
                cond += self.rng.normal(0, self.noise_level, cond.shape)

        # Stack the X, Y, and Z data to form joint observations
        data_joint = column_stack((*data, cond))

        # Create KDTree for efficient nearest neighbor search in joint space
        tree_joint = KDTree(data_joint)

        # Find k-th nearest neighbor distances in joint space
        distances, _ = tree_joint.query(data_joint, k=self.k + 1, p=self.minkowski_p)
        kth_distances = distances[:, -1]

        # Count points within k-th nearest neighbor distance in marginal spaces
        trees_marginal_cond = [KDTree(column_stack((var, cond))) for var in data]
        tree_cond = KDTree(cond)

        # Count neighbors within k-th nearest neighbor distance in X and Y spaces using
        # the maximum norm
        counts_marginal_cond = [
            [
                tree.query_ball_point(p, r=d, p=self.minkowski_p, return_length=True)
                - 1
                for p, d in zip(column_stack((var, cond)), kth_distances)
            ]
            for tree, var in zip(trees_marginal_cond, data)
        ]
        count_cond = [
            tree_cond.query_ball_point(p, r=d, p=self.minkowski_p, return_length=True)
            - 1
            for p, d in zip(cond, kth_distances)
        ]

        # Compute local CMI for each data point
        local_cmi = digamma(self.k) + array(
            [
                digamma(cz + 1) - sum(digamma(c + 1) for c in counts)
                for cz, *counts in zip(count_cond, *counts_marginal_cond)
            ]
        )

        return local_cmi / log(self.base) if self.base != "e" else local_cmi
