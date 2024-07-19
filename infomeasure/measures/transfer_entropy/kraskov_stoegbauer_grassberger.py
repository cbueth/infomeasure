"""Module for the Kraskov-Stoegbauer-Grassberger (KSG) transfer entropy estimator."""

from numpy import array, column_stack, inf
from numpy import mean as np_mean
from scipy.spatial import KDTree
from scipy.special import digamma

from ... import Config
from ...utils.types import LogBaseType
from ..base import (
    EffectiveTEMixin,
    RandomGeneratorMixin,
    TransferEntropyEstimator,
)


class KSGTEEstimator(EffectiveTEMixin, RandomGeneratorMixin, TransferEntropyEstimator):
    r"""Estimator for transfer entropy using the Kraskov-Stoegbauer-Grassberger (KSG)
    method.

    Attributes
    ----------
    source, dest : array-like
        The source and destination data used to estimate the transfer entropy.
    k : int
        Number of nearest neighbors to consider.
    tau : int
        Time delay for state space reconstruction.
    u : int
        Propagation time from when the state space reconstruction should begin.
    ds : int
        Embedding dimension for X.
    dd : int
        Embedding dimension for Y.
    noise_level : float, None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.
    minkowski_p : float, :math:`1 \leq p \leq \infty`
        The power parameter for the Minkowski metric.
        Default is np.inf for maximum norm. Use 2 for Euclidean distance.
    base : int | float | "e", optional
        The logarithm base for the transfer entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def __init__(
        self,
        source,
        dest,
        k: int = 4,
        tau: int = 1,
        u: int = 0,
        ds: int = 1,
        dd: int = 1,
        noise_level=1e-8,
        minkowski_p=inf,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the estimator with the data and parameters.

        Parameters
        ----------
        k : int
            Number of nearest neighbors to consider.
        tau : int
            Time delay for state space reconstruction.
        u : int
            Propagation time from when the state space reconstruction should begin.
        ds : int
            Embedding dimension for source.
        dd : int
            Embedding dimension for destination.
        noise_level : float, None or False
            Standard deviation of Gaussian noise to add to the data.
            Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.
        minkowski_p : float, :math:`1 \leq p \leq \infty`
            The power parameter for the Minkowski metric.
            Default is np.inf for maximum norm. Use 2 for Euclidean distance.
        """
        super().__init__(source, dest, base=base)
        self.k, self.tau, self.u = k, tau, u
        self.ds, self.dd = ds, dd
        self.noise_level = noise_level
        self.minkowski_p = minkowski_p

    def _calculate(self):
        """Calculate the transfer entropy of the data.

        Returns
        -------
        local_te : array
            Local transfer entropy for each point.
        global_te : float
            Estimated transfer entropy from X to Y.
        """

        # Ensure source and dest are numpy arrays
        source = self.source.astype(float).copy()
        dest = self.dest.astype(float).copy()

        # Add Gaussian noise to the data if the flag is set
        if self.noise_level:
            dest += self.rng.normal(0, self.noise_level, dest.shape)
            source += self.rng.normal(0, self.noise_level, source.shape)

        N = len(source)
        max_delay = max(
            self.dd * self.tau + self.u, self.ds * self.tau
        )  # maximum delay to account for all embeddings

        # Adjusted construction of multivariate data arrays with u parameter
        joint_space_data = column_stack(
            (
                dest[max_delay + self.u :],  # Adjust for u in the target series
                array(
                    [
                        dest[i - self.dd * self.tau + self.u : i + self.u : self.tau]
                        for i in range(max_delay, N - self.u)
                    ]
                ),
                # Adjust embedding of dest with u
                array(
                    [
                        source[i - self.ds * self.tau : i : self.tau]
                        for i in range(max_delay, N - self.u)
                    ]
                ),
                # Keep source embeddings aligned without u
            )
        )

        marginal_1_space_data = column_stack(
            (
                array(
                    [
                        dest[i - self.dd * self.tau + self.u : i + self.u : self.tau]
                        for i in range(max_delay, N - self.u)
                    ]
                ),
                # Adjust embedding of dest with u
                array(
                    [
                        source[i - self.ds * self.tau : i : self.tau]
                        for i in range(max_delay, N - self.u)
                    ]
                ),
                # Keep source embeddings aligned without u
            )
        )

        marginal_2_space_data = column_stack(
            (
                dest[max_delay + self.u :],  # Adjust for u in the target series
                array(
                    [
                        dest[i - self.dd * self.tau + self.u : i + self.u : self.tau]
                        for i in range(max_delay, N - self.u)
                    ]
                ),
                # Adjust embedding of dest with u
            )
        )

        # Create KDTree for efficient nearest neighbor search in joint space
        tree_joint = KDTree(joint_space_data)

        # Find distances to the k-th nearest neighbor in the joint space
        distances, _ = tree_joint.query(
            joint_space_data, k=self.k + 1, p=self.minkowski_p
        )
        kth_distances = distances[:, -1]

        # Count points for count_Y_present_past
        tree_dest_present_past = KDTree(marginal_2_space_data)
        count_dest_present_past = [
            len(tree_dest_present_past.query_ball_point(p, r=d)) - 1
            for p, d in zip(marginal_2_space_data, kth_distances)
        ]

        # Count points for count_Y_past_X_past
        tree_dest_past_source_past = KDTree(marginal_1_space_data)
        count_dest_past_source_past = [
            len(tree_dest_past_source_past.query_ball_point(p, r=d)) - 1
            for p, d in zip(marginal_1_space_data, kth_distances)
        ]

        # Count points for Count_Y_past
        data_dest_past_embedded = array(
            [dest[i - self.dd * self.tau : i : self.tau] for i in range(max_delay, N)]
        )
        tree_dest_past = KDTree(data_dest_past_embedded)
        count_dest_past = [
            len(tree_dest_past.query_ball_point(p, r=d)) - 1
            for p, d in zip(data_dest_past_embedded, kth_distances)
        ]

        # Compute local transfer entropy
        local_te = (
            digamma(self.k)
            - digamma(array(count_dest_present_past) + 1)
            - digamma(array(count_dest_past_source_past) + 1)
            + digamma(array(count_dest_past) + 1)
        )

        # Compute global transfer entropy as the mean of the local transfer entropy
        global_te = np_mean(local_te)

        return global_te, local_te
