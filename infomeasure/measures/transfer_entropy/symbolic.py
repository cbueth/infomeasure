"""Module for the Symbolic / Permutation transfer entropy estimator."""

from numpy import mean as np_mean, unique, ndarray

from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType
from ..utils.symbolic import symbolize_series
from ..utils.te_slicing import te_observations
from ..base import (
    EffectiveValueMixin,
    TransferEntropyEstimator,
)


class SymbolicTEEstimator(EffectiveValueMixin, TransferEntropyEstimator):
    r"""Estimator for the Symbolic / Permutation transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    order : int
        The size of the permutation patterns.
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

    Raises
    ------
    ValueError
        If the ``order`` is negative or not an integer.
    ValueError
        If the ``order`` is too large for the given data.
    ValueError
        If ``step_size``, ``prop_time``, and ``order`` are such that the data is too small.

    Warning
    -------
    If ``order`` is set to 1, the transfer entropy is always 0.
    """

    def __init__(
        self,
        source,
        dest,
        order: int,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with the data and parameters.

        Parameters
        ----------
        order : int
            The order of the Symbolic entropy.
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
        if not isinstance(order, int) or order < 0:
            raise ValueError("The order must be a non-negative integer.")
        if order == 1:
            logger.warning("The Symbolic mutual information is always 0 for order=1.")
        if not isinstance(step_size, int) or step_size < 0:
            raise ValueError("The step_size must be a non-negative integer.")
        if len(self.source) < (order - 1) * step_size + 1:
            raise ValueError("The data is too small for the given step_size and order.")
        self.order = order

    @staticmethod
    def _estimate_probabilities(
        symbols_source, symbols_dest, src_hist_len, dest_hist_len
    ):
        """
        Estimate the joint and conditional probabilities of the symbol sequences.

        Parameters
        ----------
        symbols_source : ndarray, (n - (order - 1) * step_size, order)
            The symbolized source data.
        symbols_dest : ndarray, (n - (order - 1) * step_size, order)
            The symbolized destination data.

        Returns
        -------
        joint_prob : dict
            Joint probabilities: p(x^l_t, y^k_t, y_{t+1})
        dest_past_prob : dict
            Embedded past destination probabilities: p(y^k_t)
        marginal_1_prob : dict
            Marginal probabilities: p(x^l_t, y^k_t)
        marginal_2_prob : dict
            Marginal probabilities: p(y^k_t, y_{t+1}).
        """
        # Slicing
        (
            joint_space_data,
            dest_past_embedded,
            marginal_1_space_data,
            marginal_2_space_data,
        ) = te_observations(
            symbols_source,
            symbols_dest,
            src_hist_len=src_hist_len,
            dest_hist_len=dest_hist_len,
            step_size=1,  # Step size is considered in symbolization
        )

        joint_unique, joint_counts = unique(
            joint_space_data, axis=0, return_counts=True
        )
        dest_past_unique, dest_past_counts = unique(
            dest_past_embedded, axis=0, return_counts=True
        )
        marginal_1_unique, marginal_1_counts = unique(
            marginal_1_space_data, axis=0, return_counts=True
        )
        marginal_2_unique, marginal_2_counts = unique(
            marginal_2_space_data, axis=0, return_counts=True
        )

        def to_prob(uniq, counts):
            total = sum(counts)
            return {
                row if isinstance(row, int) else tuple(row): count / total
                for row, count in zip(uniq, counts)
            }

        joint_prob = to_prob(joint_unique, joint_counts)
        dest_past_prob = to_prob(dest_past_unique, dest_past_counts)
        marginal_1_prob = to_prob(marginal_1_unique, marginal_1_counts)
        marginal_2_prob = to_prob(marginal_2_unique, marginal_2_counts)

        return joint_prob, dest_past_prob, marginal_1_prob, marginal_2_prob

    def _calculate(self) -> ndarray | float:
        """Calculate the Symbolic / Permutation transfer entropy.

        Returns
        -------
        local_te : array
            Local transfer entropy from X to Y for each point.
            In case of zero transfer entropy, returns 0.0.
        """

        if self.order == 1:
            return 0.0

        # Symbolize the time series dest and source, use Lehmer code
        symbols_dest = symbolize_series(
            self.dest, self.order, self.step_size, to_int=True
        )
        symbols_source = symbolize_series(
            self.source, self.order, self.step_size, to_int=True
        )

        # Estimate joint and conditional probabilities
        joint_prob, dest_past_prob, marginal_1_prob, marginal_2_prob = (
            self._estimate_probabilities(
                symbols_source,
                symbols_dest,
                src_hist_len=self.src_hist_len,
                dest_hist_len=self.dest_hist_len,
            )
        )

        # Calculate Transfer Entropy for each permutation pattern
        te_perm = []
        for pattern in joint_prob:
            p_joint = joint_prob[pattern]  # p(x^l_t, y^k_t, y_{t+1})

            # Define conditional patterns
            cond_pattern_joint = pattern[
                : self.src_hist_len + self.dest_hist_len
            ]  # (x^l_t, y^k_t)
            cond_pattern_marginal = pattern[
                self.src_hist_len : self.src_hist_len + self.dest_hist_len
            ]  # y^k_t
            cond_pattern_conditional = pattern[self.src_hist_len :]  # (y^k_t, y_{t+1})

            # Retrieve probabilities from the precomputed dictionaries
            p_cond_joint = marginal_1_prob.get(cond_pattern_joint, 0)  # p(x^l_t, y^k_t)
            p_cond_marginal = dest_past_prob.get(cond_pattern_marginal, 0)  # p(y^k_t)
            p_cond_conditional = marginal_2_prob.get(
                cond_pattern_conditional, 0
            )  # p(y^k_t, y_{t+1})

            # Compute the conditional probabilities
            p_conditional_joint = (
                p_joint / p_cond_joint if p_cond_joint > 0 else 0
            )  # p(y_{t+1} | x^l_t, y^k_t)
            p_conditional_marginal = (
                p_cond_conditional / p_cond_marginal if p_cond_marginal > 0 else 0
            )  # p(y_{t+1} | y^k_t)

            if p_joint > 0 and p_conditional_joint > 0 and p_conditional_marginal > 0:
                # Using the TE formula
                te_perm.append(
                    self._log_base(p_conditional_joint / p_conditional_marginal)
                    * p_joint
                )
        if len(te_perm) == 0:
            return 0.0
        # Compute average of Local Transfer Entropy values
        return np_mean(te_perm)
