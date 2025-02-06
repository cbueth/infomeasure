"""Module for the discrete transfer entropy estimator."""

from ... import Config
from ...utils.types import LogBaseType
from ..base import EffectiveValueMixin, TransferEntropyEstimator
from ..entropy.discrete import DiscreteEntropyEstimator


class DiscreteTEEstimator(EffectiveValueMixin, TransferEntropyEstimator):
    """Estimator for discrete transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and destination (Y) data used to estimate the transfer entropy.
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
        *,  # all following parameters are keyword-only
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with the data and parameters.

        Parameters
        ----------
        src_hist_len, dest_hist_len : int
            Embedding lengths for the source and destination variables.
        delay : int
            Time delay between the source and destination variables.
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

    def _calculate(self):
        """Estimate the Discrete Transfer Entropy.

        Returns
        -------
        float
            The Renyi transfer entropy.
        """

        return self._generic_te_from_entropy(
            estimator=DiscreteEntropyEstimator,
            kwargs=dict(base=self.base),
        )
