"""Module for the discrete transfer entropy estimator."""

from numpy import log

from ... import Config
from ...utils.types import LogBaseType
from ..base import LogBaseMixin, TransferEntropyEstimator


class DiscreteTEEstimator(LogBaseMixin, TransferEntropyEstimator):
    """Estimator for discrete transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source and destination data used to estimate the transfer entropy.
    l, k : int
        Embedding lengths for the source and destination variables.
    delay : int
        Time delay between the source and destination variables.
    base : int | float | "e", optional
        The logarithm base for the transfer entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Methods
    -------
    calculate()
        Calculate the transfer entropy from source to destination.
    """

    def __init__(
        self, source, dest, l, k, delay, base: LogBaseType = Config.get("base")
    ):
        """Initialize the estimator with the data and parameters.

        Parameters
        ----------
        l, k : int
            Embedding lengths for the source and destination variables.
        delay : int
            Time delay between the source and destination variables.
        """
        super().__init__(source, dest, base=base)
        self.l = l
        self.k = k
        self.delay = delay

    def calculate(self):
        """Calculate the transfer entropy of the data.

        Returns
        -------
        float
            The calculated transfer entropy.
        """

        (
            source_next_past_count,
            source_past_count,
            next_past_count,
            past_count,
            observations,
        ) = count_tuples(self.source, self.dest, self.l, self.k, self.delay)

        te = 0
        for (s_t, d_t, d_t_k), p_s_t_d_t_d_t_k in source_next_past_count.items():
            p_s_t_d_t_d_t_k /= observations

            p_s_t_d_t_k = source_past_count[s_t, d_t_k] / observations
            p_d_t_d_t_k = next_past_count[d_t, d_t_k] / past_count[d_t_k]
            p_d_t_k = past_count[d_t_k] / observations

            log_term = (p_d_t_d_t_k / p_d_t_k) / (p_s_t_d_t_k / p_d_t_k)
            local_value = log(log_term)

            te += p_s_t_d_t_d_t_k * local_value

        # Convert to the base of choice
        if self.base != "e":
            te /= log(self.base)

        return te


def count_tuples(source, dest, l, k, delay):
    """
    Count tuples for Transfer Entropy computation.

    Parameters
    ----------
    source, dest : array-like
        Source and destination time-series data.
    l, k : int
        Embedding lengths for the source and destination variables.
    delay : int
        Time delay between the source and destination variables.

    Returns
    -------
    source_next_past_count : dict
        Count for source, next state of destination, and past state of destination.
    source_past_count : dict
        Count for source and past state of destination.
    next_past_count : dict
        Count for next state and past state of destination.
    past_count : dict
        Count for past state of destination.
    observations : int
        Total number of observations.
    """
    source_next_past_count = {}
    source_past_count = {}
    next_past_count = {}
    past_count = {}
    observations = 0

    for t in range(max(k, l + delay), len(dest)):
        # Next state for the destination variable
        next_state_dest = dest[t]

        # Update past states
        past_state_dest = dest[t - k : t]
        past_state_source = source[t - delay - l + 1 : t - delay + 1]

        # Convert arrays to tuple to use as dictionary keys
        past_state_dest_t = tuple(past_state_dest)
        past_state_source_t = tuple(past_state_source)

        # Update counts
        if (
            past_state_source_t,
            next_state_dest,
            past_state_dest_t,
        ) in source_next_past_count:
            source_next_past_count[
                past_state_source_t, next_state_dest, past_state_dest_t
            ] += 1
        else:
            source_next_past_count[
                past_state_source_t, next_state_dest, past_state_dest_t
            ] = 1

        if (past_state_source_t, past_state_dest_t) in source_past_count:
            source_past_count[past_state_source_t, past_state_dest_t] += 1
        else:
            source_past_count[past_state_source_t, past_state_dest_t] = 1

        if (next_state_dest, past_state_dest_t) in next_past_count:
            next_past_count[next_state_dest, past_state_dest_t] += 1
        else:
            next_past_count[next_state_dest, past_state_dest_t] = 1

        if past_state_dest_t in past_count:
            past_count[past_state_dest_t] += 1
        else:
            past_count[past_state_dest_t] = 1

        observations += 1

    return (
        source_next_past_count,
        source_past_count,
        next_past_count,
        past_count,
        observations,
    )
