"""Module for the Symbolic / Permutation transfer entropy estimator."""

from collections import Counter

from numpy import argsort, mean as np_mean, array

from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType
from ..base import (
    EffectiveTEMixin,
    TransferEntropyEstimator,
)


class SymbolicTEEstimator(EffectiveTEMixin, TransferEntropyEstimator):
    r"""Estimator for the Symbolic / Permutation transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source and dest data used to estimate the transfer entropy.
    order : int
        The order of the Symbolic entropy.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
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
        If ``step_size``, ``offset``, and ``order`` are such that the data is too small.

    Warning
    -------
    If ``order`` is set to 1, the transfer entropy is always 0.
    """

    def __init__(
        self,
        source,
        dest,
        order: int,
        offset: int = 0,
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
            offset=offset,
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

    def _calculate(self) -> tuple:
        """Calculate the Symbolic / Permutation transfer entropy.

        Returns
        -------
        global_te : float
            Estimated transfer entropy from X to Y.
        local_te : array
            Local transfer entropy for each point.
        """

        if self.order == 1:
            return 0.0, array([])

        def _get_pattern_type(subsequence):
            """
            Determine the permutation pattern type of a given subsequence.

            Parameters:
            subsequence (list or array): A subsequence of the time series.

            Returns:
            tuple: A tuple representing the permutation pattern type.
            """
            return tuple(argsort(subsequence))

        def _symbolize_series(series, order, step_size):
            """
            Convert a time series into a sequence of symbols (permutation patterns).

            Parameters:
            series (list or array): The time series to be symbolized.
            step_size (int): The time delay (l).

            Returns:
            list: A list of tuples representing the symbolized series.
            """
            T = len(series)  # Length of the time series
            patterns = []
            for i in range(T - (order - 1) * step_size):  # Iterate over time series
                subsequence = [
                    series[i + j * step_size] for j in range(order)
                ]  # Extract subsequence
                pattern = _get_pattern_type(subsequence)  # Determine pattern type
                patterns.append(pattern)  # Append pattern to list
            return patterns

        def _estimate_probabilities(joint_sample_space):
            """
            Estimate the joint and conditional probabilities of the symbol sequences.

            Parameters:
            joint_sample_space (list): A list of tuples representing the joint sample space.

            Returns:
            dict: Joint probabilities: p(y_{t+1}, y^k_t, x^l_t)
            dict: Joint probabilities: p(y^k_t, x^l_t)
            dict: marginal probabilities: p(x^l_t)
            dict: joint probabilities:  p(y_{t+1}, y^k_t).
            """
            joint_counts = Counter()  # Counter for (y_{n+1}, y^{(k)}_n, x^{(l)}_n)
            cond_counts_joint = Counter()  # Counter for (y^{(k)}_n, x^{(l)}_n)
            cond_counts_marginal = Counter()  # Counter for (x^{(l)}_n)
            cond_counts_conditional = Counter()  # Counter for (y_{n+1}, y^{(k)}_n)

            for joint_pattern in joint_sample_space:
                # Define patterns
                cond_pattern_joint = (
                    joint_pattern[1],
                    joint_pattern[2],
                )  # (y^{(k)}_n, x^{(l)}_n)
                cond_pattern_marginal = joint_pattern[1]  # y^{(k)}_n
                cond_pattern_conditional = (
                    joint_pattern[0],
                    joint_pattern[1],
                )  # (y_{n+1}, y^{(k)}_n)

                # Update counts
                joint_counts[joint_pattern] += 1
                cond_counts_joint[cond_pattern_joint] += 1
                cond_counts_marginal[cond_pattern_marginal] += 1
                cond_counts_conditional[cond_pattern_conditional] += 1

            # Calculate total counts
            joint_total = sum(joint_counts.values())
            cond_total_joint = sum(cond_counts_joint.values())
            cond_total_marginal = sum(cond_counts_marginal.values())
            cond_total_conditional = sum(cond_counts_conditional.values())

            # Calculate probabilities
            joint_prob = {k: v / joint_total for k, v in joint_counts.items()}
            cond_prob_joint = {
                k: v / cond_total_joint for k, v in cond_counts_joint.items()
            }
            cond_prob_marginal = {
                k: v / cond_total_marginal for k, v in cond_counts_marginal.items()
            }
            cond_prob_conditional = {
                k: v / cond_total_conditional
                for k, v in cond_counts_conditional.items()
            }

            return (
                joint_prob,
                cond_prob_joint,
                cond_prob_marginal,
                cond_prob_conditional,
            )

        # Symbolize the time series dest and source
        symbols_dest = _symbolize_series(self.dest, self.order, self.step_size)
        symbols_source = _symbolize_series(self.source, self.order, self.step_size)

        # Create joint sample space
        joint_sample_space = [
            (
                symbols_dest[i],
                tuple(symbols_dest[i - self.dest_hist_len : i]),
                tuple(symbols_source[i - self.src_hist_len : i]),
            )
            for i in range(
                max(self.src_hist_len, self.dest_hist_len), len(symbols_dest)
            )
        ]

        # Estimate joint and conditional probabilities
        joint_prob, cond_prob_joint, cond_prob_marginal, cond_prob_conditional = (
            _estimate_probabilities(joint_sample_space)
        )

        # Calculate Local Transfer Entropy
        local_te = []
        for pattern in joint_prob:
            p_joint = joint_prob[pattern]  # p(y_{i+u}, y_i, x_i)

            # Define conditional patterns
            cond_pattern_joint = (pattern[1], pattern[2])  # (y_i, x_i)
            cond_pattern_marginal = pattern[1]  # y_i
            cond_pattern_conditional = (pattern[0], pattern[1])  # (y_{i+u}, y_i)

            # Retrieve probabilities from the precomputed dictionaries
            p_cond_joint = cond_prob_joint.get(cond_pattern_joint, 0)  # p(y_i, x_i)
            p_cond_marginal = cond_prob_marginal.get(cond_pattern_marginal, 0)  # p(y_i)
            p_cond_conditional = cond_prob_conditional.get(
                cond_pattern_conditional, 0
            )  # p(y_{i+u}, y_i)

            # Compute the conditional probabilities
            p_conditional_joint = (
                p_joint / p_cond_joint if p_cond_joint > 0 else 0
            )  # p(y_{i+u} | y_i, x_i)
            p_conditional_marginal = (
                p_cond_conditional / p_cond_marginal if p_cond_marginal > 0 else 0
            )  # p(y_{i+u} | y_i)

            if p_joint > 0 and p_conditional_joint > 0 and p_conditional_marginal > 0:
                # Using the TE formula
                local_te_value = self._log_base(
                    p_conditional_joint / p_conditional_marginal
                )
                local_te.extend([local_te_value] * int(p_joint * len(symbols_dest)))
        if len(local_te) == 0:
            return 0.0, array([])
        # Compute average and standard deviation of Local Transfer Entropy values
        average_te = np_mean(local_te)

        return average_te, array(local_te)
