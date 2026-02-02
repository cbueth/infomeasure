"""Module for the Kraskov-Stoegbauer-Grassberger (KSG) transfer entropy estimator."""

from abc import ABC

from numpy import array, inf, ndarray, log, nextafter
from scipy.spatial import KDTree
from scipy.special import digamma

from ..base import (
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
    ksg_id : int, default=1
        The KSG estimator variant to use (1 or 2).
        Type I uses strict inequality for neighbour counting and the corresponding
        formula. Type II uses non-strict inequality and a different formula.
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
        ksg_id: int = 1,
        noise_level=1e-8,
        minkowski_p=inf,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        offset: int = None,
        base: LogBaseType = Config.get("base"),
        **kwargs,
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
        ksg_id : int, default=1
            The KSG estimator variant to use (1 or 2).
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
                **kwargs,
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
                **kwargs,
            )
        self.k = k
        if ksg_id not in {1, 2}:
            raise ValueError(f"ksg_id must be 1 or 2, but got {ksg_id}.")
        self.ksg_id = ksg_id
        self.noise_level = noise_level
        self.minkowski_p = minkowski_p


class KSGTEEstimator(BaseKSGTEEstimator, TransferEntropyEstimator):
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

        # Count points in marginal spaces
        tree_dest_past_present = KDTree(marginal_2_space_data)
        tree_source_past_dest_past = KDTree(marginal_1_space_data)
        tree_dest_past = KDTree(data_dest_past_embedded)

        if self.ksg_id == 1:
            r_strict = nextafter(kth_distances, -inf)
            count_dest_past_present = tree_dest_past_present.query_ball_point(
                marginal_2_space_data,
                r=r_strict,
                p=self.minkowski_p,
                return_length=True,
            ) - (kth_distances > 0).astype(int)
            count_source_past_dest_past = tree_source_past_dest_past.query_ball_point(
                marginal_1_space_data,
                r=r_strict,
                p=self.minkowski_p,
                return_length=True,
            ) - (kth_distances > 0).astype(int)
            count_dest_past = tree_dest_past.query_ball_point(
                data_dest_past_embedded,
                r=r_strict,
                p=self.minkowski_p,
                return_length=True,
            ) - (kth_distances > 0).astype(int)

            # Compute local transfer entropy
            local_te = (
                digamma(self.k)
                - digamma(array(count_dest_past_present) + 1)
                - digamma(array(count_source_past_dest_past) + 1)
                + digamma(array(count_dest_past) + 1)
            )
        else:
            count_dest_past_present = tree_dest_past_present.query_ball_point(
                marginal_2_space_data,
                r=kth_distances,
                p=self.minkowski_p,
                return_length=True,
            )
            count_source_past_dest_past = tree_source_past_dest_past.query_ball_point(
                marginal_1_space_data,
                r=kth_distances,
                p=self.minkowski_p,
                return_length=True,
            )
            count_dest_past = tree_dest_past.query_ball_point(
                data_dest_past_embedded,
                r=kth_distances,
                p=self.minkowski_p,
                return_length=True,
            )

            # Compute local transfer entropy
            local_te = (
                digamma(self.k)
                - 1.0 / self.k
                - digamma(array(count_dest_past_present))
                - digamma(array(count_source_past_dest_past))
                + digamma(array(count_dest_past))
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
    The estimator supports two variants:

    - **Type I** (``ksg_id=1``): Uses strict inequality for counting neighbors in
      marginal spaces (dist < eps).
    - **Type II** (``ksg_id=2``): Uses non-strict inequality (dist <= eps) and a
      modified formula.

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

        # Count points in marginal spaces
        tree_cond_dest_past_present = KDTree(marginal_2_space_data)
        tree_source_past_cond_dest_past = KDTree(marginal_1_space_data)
        tree_dest_past_cond = KDTree(data_dest_past_embedded)

        if self.ksg_id == 1:
            r_strict = nextafter(kth_distances, -inf)
            count_cond_dest_past_present = tree_cond_dest_past_present.query_ball_point(
                marginal_2_space_data,
                r=r_strict,
                p=self.minkowski_p,
                return_length=True,
            ) - (kth_distances > 0).astype(int)
            count_source_past_cond_dest_past = (
                tree_source_past_cond_dest_past.query_ball_point(
                    marginal_1_space_data,
                    r=r_strict,
                    p=self.minkowski_p,
                    return_length=True,
                )
                - (kth_distances > 0).astype(int)
            )
            count_dest_past_cond = tree_dest_past_cond.query_ball_point(
                data_dest_past_embedded,
                r=r_strict,
                p=self.minkowski_p,
                return_length=True,
            ) - (kth_distances > 0).astype(int)

            # Compute local conditional transfer entropy
            local_cte = (
                digamma(self.k)
                - digamma(array(count_cond_dest_past_present) + 1)
                - digamma(array(count_source_past_cond_dest_past) + 1)
                + digamma(array(count_dest_past_cond) + 1)
            )
        else:
            count_cond_dest_past_present = tree_cond_dest_past_present.query_ball_point(
                marginal_2_space_data,
                r=kth_distances,
                p=self.minkowski_p,
                return_length=True,
            )
            count_source_past_cond_dest_past = (
                tree_source_past_cond_dest_past.query_ball_point(
                    marginal_1_space_data,
                    r=kth_distances,
                    p=self.minkowski_p,
                    return_length=True,
                )
            )
            count_dest_past_cond = tree_dest_past_cond.query_ball_point(
                data_dest_past_embedded,
                r=kth_distances,
                p=self.minkowski_p,
                return_length=True,
            )

            # Compute local conditional transfer entropy
            local_cte = (
                digamma(self.k)
                - 1.0 / self.k
                - digamma(array(count_cond_dest_past_present))
                - digamma(array(count_source_past_cond_dest_past))
                + digamma(array(count_dest_past_cond))
            )
        local_cte = array(local_cte)

        return local_cte / log(self.base) if self.base != "e" else local_cte
