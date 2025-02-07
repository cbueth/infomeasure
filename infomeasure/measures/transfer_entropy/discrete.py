"""Module for the discrete transfer entropy estimator."""

from abc import ABC

from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType
from ..base import (
    EffectiveValueMixin,
    TransferEntropyEstimator,
    ConditionalTransferEntropyEstimator,
)
from ..entropy.discrete import DiscreteEntropyEstimator


class BaseDiscreteTEEstimator(ABC):
    """Base class for discrete transfer entropy estimators.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and destination (Y) data used to estimate the transfer entropy.
    cond : array-like, optional
        The conditional data used to estimate the conditional transfer entropy.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Assumed time taken by info to transfer from source to destination.
        Not compatible with the ``cond`` parameter / conditional TE.
    step_size : int, optional
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int, optional
        Number of past observations to consider for the source and destination data.
    cond_hist_len : int, optional
        Number of past observations to consider for the conditional data.
        Only used for conditional transfer entropy.
    base : int | float | "e", optional
        The logarithm base for the transfer entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def __init__(
        self,
        source,
        dest,
        cond=None,
        *,  # all following parameters are keyword-only
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the BaseDiscreteTEEstimator.

        Parameters
        ----------
        source, dest : array-like
            The source (X) and destination (Y) data used to estimate the transfer entropy.
        cond : array-like, optional
            The conditional data used to estimate the conditional transfer entropy.
        prop_time : int, optional
            Number of positions to shift the data arrays relative to each other (multiple of
            ``step_size``).
            Delay/lag/shift between the variables, representing propagation time.
            Assumed time taken by info to transfer from source to destination
            Not compatible with the ``cond`` parameter / conditional TE.
        step_size : int, optional
            Step size between elements for the state space reconstruction.
        src_hist_len, dest_hist_len : int, optional
            Number of past observations to consider for the source and destination data.
        cond_hist_len : int, optional
            Number of past observations to consider for the conditional data.
            Only used for conditional transfer entropy.
        base : int | float | "e", optional
            The logarithm base for the transfer entropy calculation.
            The default can be set
            with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
        """
        self.source = source
        self.dest = dest
        if cond is None:
            super().__init__(
                source,
                dest,
                prop_time=prop_time,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                step_size=step_size,
                base=base,
            )
        else:
            super().__init__(
                source,
                dest,
                cond,
                step_size=step_size,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                cond_hist_len=cond_hist_len,
                prop_time=prop_time,
                base=base,
            )
            if (
                self.source.dtype.kind == "f"
                or self.dest.dtype.kind == "f"
                or (self.cond is not None and self.cond.dtype.kind == "f")
            ):
                logger.warning(
                    "The data looks like a float array ("
                    f"source: {self.source.dtype}, dest: {self.dest.dtype}). "
                    "Make sure the data is properly symbolized or discretized "
                    "for the transfer entropy estimation."
                )
            if hasattr(self, "cond") and self.cond.dtype.kind == "f":
                logger.warning(
                    "The conditional data looks like a float array ("
                    f"{self.cond.dtype}). "
                    "Make sure the data is properly symbolized or discretized "
                    "for the conditional transfer entropy estimation."
                )


class DiscreteTEEstimator(
    BaseDiscreteTEEstimator, EffectiveValueMixin, TransferEntropyEstimator
):
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

    def _calculate(self):
        """Estimate the Discrete Transfer Entropy."""

        return self._generic_te_from_entropy(
            estimator=DiscreteEntropyEstimator, kwargs=dict(base=self.base)
        )


class DiscreteCTEEstimator(
    BaseDiscreteTEEstimator, ConditionalTransferEntropyEstimator
):
    """Estimator for discrete conditional transfer entropy.

    Attributes
    ----------
    source, dest, cond : array-like
        The source (X), destination (Y), and conditional (Z) data used to estimate the
        conditional transfer entropy.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len, cond_hist_len : int, optional
        Number of past observations to consider for the source, destination,
        and conditional data.
    prop_time : int, optional
        Not compatible with the ``cond`` parameter / conditional TE.
    base : int | float | "e", optional
        The logarithm base for the transfer entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def _calculate(self):
        """Estimate the Discrete Conditional Transfer Entropy.

        Returns
        -------
        float
            The Renyi conditional transfer entropy.
        """
        return self._generic_cte_from_entropy(
            estimator=DiscreteEntropyEstimator, kwargs=dict(base=self.base)
        )
