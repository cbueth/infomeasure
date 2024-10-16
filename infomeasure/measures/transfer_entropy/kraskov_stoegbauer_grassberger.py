"""Module for the Kraskov-Stoegbauer-Grassberger (KSG) transfer entropy estimator."""

from numpy import array, inf
from numpy import mean as np_mean
from scipy.spatial import KDTree
from scipy.special import digamma

from ..utils.te_slicing import te_observations
from ... import Config
from ...utils.types import LogBaseType
from ..base import (
    EffectiveTEMixin,
    TransferEntropyEstimator,
)


class KSGTEEstimator(EffectiveTEMixin, TransferEntropyEstimator):
    r"""Estimator for transfer entropy using the Kraskov-Stoegbauer-Grassberger (KSG)
    method.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and destination (Y) data used to estimate the transfer entropy.
    k : int
        Number of nearest neighbors to consider.
    noise_level : float, None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.
    minkowski_p : float, :math:`1 \leq p \leq \infty`
        The power parameter for the Minkowski metric.
        Default is np.inf for maximum norm. Use 2 for Euclidean distance.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Assumed time taken by info to transfer from source to destination.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.
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
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        noise_level=1e-8,
        minkowski_p=inf,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the estimator with the data and parameters.

        Parameters
        ----------
        k : int
            Number of nearest neighbors to consider.
        noise_level : float, None or False
            Standard deviation of Gaussian noise to add to the data.
            Adds :math:`\mathcal{N}(0, \text{noise}^2)` to each data point.
        minkowski_p : float, :math:`1 \leq p \leq \infty`
            The power parameter for the Minkowski metric.
            Default is np.inf for maximum norm. Use 2 for Euclidean distance.
        """
        super().__init__(
            source,
            dest,
            prop_time=prop_time,
            step_size=step_size,
            src_hist_len=src_hist_len,
            dest_hist_len=dest_hist_len,
            base=base,
        )
        self.k = k
        self.noise_level = noise_level
        self.minkowski_p = minkowski_p

    def _calculate(self):
        """Calculate the transfer entropy of the data.

        Returns
        -------
        global_te : float
            Estimated transfer entropy from X to Y.
        local_te : array
            Local transfer entropy for each point.
        """

        # Ensure source and dest are numpy arrays
        source = self.source.astype(float).copy()
        dest = self.dest.astype(float).copy()

        # Add Gaussian noise to the data if the flag is set
        if self.noise_level:
            dest += self.rng.normal(0, self.noise_level, dest.shape)
            source += self.rng.normal(0, self.noise_level, source.shape)

        (
            joint_space_data,
            data_dest_past_embedded,
            marginal_1_space_data,
            marginal_2_space_data,
        ) = te_observations(
            source,
            dest,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            step_size=self.step_size,
            permute_src=self.permute_src,
        )

        # Create KDTree for efficient nearest neighbor search in joint space
        tree_joint = KDTree(joint_space_data)

        # Find distances to the k-th nearest neighbor in the joint space
        distances, _ = tree_joint.query(
            joint_space_data, k=self.k + 1, p=self.minkowski_p
        )
        kth_distances = distances[:, -1]  # get last column with k-th distances

        # Count points for count_Y_past_present
        tree_dest_past_present = KDTree(marginal_2_space_data)
        count_dest_past_present = [
            len(tree_dest_past_present.query_ball_point(p, r=d)) - 1
            for p, d in zip(marginal_2_space_data, kth_distances)
        ]
        # Count points for count_X_past_Y_past
        tree_source_past_dest_past = KDTree(marginal_1_space_data)
        count_source_past_dest_past = [
            len(tree_source_past_dest_past.query_ball_point(p, r=d)) - 1
            for p, d in zip(marginal_1_space_data, kth_distances)
        ]
        # Count points for Count_Y_past
        tree_dest_past = KDTree(data_dest_past_embedded)
        count_dest_past = [
            len(tree_dest_past.query_ball_point(p, r=d)) - 1
            for p, d in zip(data_dest_past_embedded, kth_distances)
        ]

        # Compute local transfer entropy
        local_te = (
            digamma(self.k)
            - digamma(array(count_dest_past_present) + 1)
            - digamma(array(count_source_past_dest_past) + 1)
            + digamma(array(count_dest_past) + 1)
        )

        # Compute global transfer entropy as the mean of the local transfer entropy
        global_te = np_mean(local_te)

        return global_te, local_te
