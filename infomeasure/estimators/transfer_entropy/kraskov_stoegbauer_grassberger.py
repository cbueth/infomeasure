"""Module for the Kraskov-Stoegbauer-Grassberger (KSG) transfer entropy estimator."""

from abc import ABC

from numpy import array, inf, ndarray, log
from scipy.spatial import KDTree
from scipy.special import digamma

from ..base import (
    PValueMixin,
    EffectiveValueMixin,
    TransferEntropyEstimator,
    ConditionalTransferEntropyEstimator,
)
from ..utils.te_slicing import te_observations, cte_observations
from ... import Config
from ...utils.types import LogBaseType


class BaseKSGTEEstimator(ABC):
    r"""Base class for transfer entropy using the Kraskov-Stoegbauer-Grassberger (KSG)
    method.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and destination (Y) data used to estimate the transfer entropy.
    cond
        The conditional data used to estimate the conditional transfer entropy.
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
        Not compatible with the ``cond`` parameter / conditional TE.
        Alternatively called `offset`.
    step_size : int, optional
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int, optional
        Number of past observations to consider for the source and destination data.
    cond_hist_len : int, optional
        Number of past observations to consider for the conditional data.
        Only used for conditional transfer entropy.
    """

    def __init__(
        self,
        source,
        dest,
        *,  # Enforce keyword-only arguments
        cond=None,
        k: int = 4,
        noise_level=1e-8,
        minkowski_p=inf,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        offset: int = None,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the BaseKSGTEEstimator.

        Parameters
        ----------
        source, dest : array-like
            The source (X) and destination (Y) data used to estimate the transfer entropy.
        cond
            The conditional data used to estimate the conditional transfer entropy.
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
            Not compatible with the ``cond`` parameter / conditional TE.
            Alternatively called `offset`.
        step_size : int, optional
            Step size between elements for the state space reconstruction.
        src_hist_len, dest_hist_len : int, optional
            Number of past observations to consider for the source and destination data.
        cond_hist_len : int, optional
            Number of past observations to consider for the conditional data.
            Only used for conditional transfer entropy.
        """
        if cond is None:
            super().__init__(
                source,
                dest,
                prop_time=prop_time,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                step_size=step_size,
                offset=offset,
                base=base,
            )
        else:
            super().__init__(
                source,
                dest,
                cond=cond,
                step_size=step_size,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                cond_hist_len=cond_hist_len,
                prop_time=prop_time,
                offset=offset,
                base=base,
            )
        self.k = k
        self.noise_level = noise_level
        self.minkowski_p = minkowski_p


class KSGTEEstimator(
    BaseKSGTEEstimator, PValueMixin, EffectiveValueMixin, TransferEntropyEstimator
):
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
        Alternatively called `offset`.
    step_size : int, optional
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.

    Notes
    -----
    Changing the number of nearest neighbors ``k`` can change the outcome,
    but the default value of :math:`k=4` is recommended by :cite:p:`miKSG2004`.
    """

    def _calculate(self) -> ndarray:
        """Calculate the transfer entropy of the data.

        Returns
        -------
        local_te : array
            Local transfer entropy from X to Y for each point.
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
            resample_src=self.resample_src,
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
            tree_dest_past_present.query_ball_point(p, r=d, return_length=True) - 1
            for p, d in zip(marginal_2_space_data, kth_distances)
        ]
        # Count points for count_X_past_Y_past
        tree_source_past_dest_past = KDTree(marginal_1_space_data)
        count_source_past_dest_past = [
            tree_source_past_dest_past.query_ball_point(p, r=d, return_length=True) - 1
            for p, d in zip(marginal_1_space_data, kth_distances)
        ]
        # Count points for Count_Y_past
        tree_dest_past = KDTree(data_dest_past_embedded)
        count_dest_past = [
            tree_dest_past.query_ball_point(p, r=d, return_length=True) - 1
            for p, d in zip(data_dest_past_embedded, kth_distances)
        ]

        # Compute local transfer entropy
        local_te = (
            digamma(self.k)
            - digamma(array(count_dest_past_present) + 1)
            - digamma(array(count_source_past_dest_past) + 1)
            + digamma(array(count_dest_past) + 1)
        )

        return local_te / log(self.base) if self.base != "e" else local_te


class KSGCTEEstimator(BaseKSGTEEstimator, ConditionalTransferEntropyEstimator):
    """Estimator for conditional transfer entropy using the
    Kraskov-Stoegbauer-Grassberger (KSG) method.

    Attributes
    ----------
    source, dest, cond : array-like
        The source (X), destination (Y), and conditional (Z) data used to estimate the
        conditional transfer entropy.
    k : int
        Number of nearest neighbors to consider.
    noise_level : float, None or False
        Standard deviation of Gaussian noise to add to the data.
        Adds :math:`\\mathcal{N}(0, \text{noise}^2)` to each data point.
    minkowski_p : float, :math:`1 \\leq p \\leq \\infty`
        The power parameter for the Minkowski metric.
        Default is np.inf for maximum norm. Use 2 for Euclidean distance.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len, cond_hist_len : int, optional
        Number of past observations to consider for the source, destination,
        and conditional data.
    prop_time : int, optional
        Not compatible with the ``cond`` parameter / conditional TE.

    Notes
    -----
    Changing the number of nearest neighbors ``k`` can change the outcome,
    but the default value of :math:`k=4` is recommended by :cite:p:`miKSG2004`.
    """

    def _calculate(self) -> ndarray:
        """Calculate the conditional transfer entropy of the data.

        Returns
        -------
        local_cte : array
            Local conditional transfer entropy from X to Y given Z for each point.
        """

        # Ensure source, dest, and cond are numpy arrays
        source = self.source.astype(float).copy()
        dest = self.dest.astype(float).copy()
        cond = self.cond.astype(float).copy()

        # Add Gaussian noise to the data if the flag is set
        if self.noise_level:
            dest += self.rng.normal(0, self.noise_level, dest.shape)
            source += self.rng.normal(0, self.noise_level, source.shape)
            cond += self.rng.normal(0, self.noise_level, cond.shape)

        (
            joint_space_data,
            data_dest_past_embedded,
            marginal_1_space_data,
            marginal_2_space_data,
        ) = cte_observations(
            source,
            dest,
            cond,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            cond_hist_len=self.cond_hist_len,
            step_size=self.step_size,
        )

        # Create KDTree for efficient nearest neighbor search in joint space
        tree_joint = KDTree(joint_space_data)

        # Find distances to the k-th nearest
        distances, _ = tree_joint.query(
            joint_space_data, k=self.k + 1, p=self.minkowski_p
        )
        kth_distances = distances[:, -1]

        # Count points for count_cond_Y_past_present
        tree_cond_dest_past_present = KDTree(marginal_2_space_data)
        count_cond_dest_past_present = [
            tree_cond_dest_past_present.query_ball_point(p, r=d, return_length=True) - 1
            for p, d in zip(marginal_2_space_data, kth_distances)
        ]
        # Count points for count_X_past_cond_Y_past
        tree_source_past_cond_dest_past = KDTree(marginal_1_space_data)
        count_source_past_cond_dest_past = [
            tree_source_past_cond_dest_past.query_ball_point(p, r=d, return_length=True)
            - 1
            for p, d in zip(marginal_1_space_data, kth_distances)
        ]
        # Count points for Count_Y_past_cond
        tree_dest_past_cond = KDTree(data_dest_past_embedded)
        count_dest_past_cond = [
            tree_dest_past_cond.query_ball_point(p, r=d, return_length=True) - 1
            for p, d in zip(data_dest_past_embedded, kth_distances)
        ]

        # Compute local conditional transfer entropy
        local_cte = (
            digamma(self.k)
            - digamma(array(count_cond_dest_past_present) + 1)
            - digamma(array(count_source_past_cond_dest_past) + 1)
            + digamma(array(count_dest_past_cond) + 1)
        )

        return local_cte / log(self.base) if self.base != "e" else local_cte
