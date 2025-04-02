"""Module for the discrete transfer entropy estimator."""

from abc import ABC

from numpy import ndarray

from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType
from ..base import (
    ConditionalTransferEntropyEstimator,
    EffectiveValueMixin,
    PValueMixin,
    TransferEntropyEstimator,
)
from ..utils.discrete_transfer_entropy import combined_te_form
from ..utils.te_slicing import cte_observations, te_observations


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
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        offset: int = None,
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
            Alternatively called `offset`.
        step_size : int, optional
            Step size between elements for the state space reconstruction.
        src_hist_len, dest_hist_len : int, optional
            Number of past observations to consider for the source and destination data.
        cond_hist_len : int, optional
            Number of past observations to consider for the conditional data.
            Only used for conditional transfer entropy.
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
    BaseDiscreteTEEstimator, PValueMixin, EffectiveValueMixin, TransferEntropyEstimator
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
        Alternatively called `offset`.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.
    """

    def _calculate(self):
        """Estimate the Discrete Transfer Entropy."""
        return combined_te_form(
            te_observations,
            self.source,
            self.dest,
            local=False,
            log_func=self._log_base,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            step_size=self.step_size,
            permute_src=self.permute_src,
            resample_src=self.resample_src,
        )

    def _extract_local_values(self) -> ndarray:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of cmi.
        """
        return combined_te_form(
            te_observations,
            self.source,
            self.dest,
            local=True,
            log_func=self._log_base,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            step_size=self.step_size,
            permute_src=self.permute_src,
            resample_src=self.resample_src,
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
    """

    def _calculate(self):
        """Estimate the Discrete Transfer Entropy."""
        return combined_te_form(
            cte_observations,
            self.source,
            self.dest,
            self.cond,
            local=False,
            log_func=self._log_base,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            cond_hist_len=self.cond_hist_len,
            step_size=self.step_size,
        )

    def _extract_local_values(self) -> ndarray:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of cmi.
        """
        return combined_te_form(
            cte_observations,
            self.source,
            self.dest,
            self.cond,
            local=True,
            log_func=self._log_base,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            cond_hist_len=self.cond_hist_len,
            step_size=self.step_size,
        )
