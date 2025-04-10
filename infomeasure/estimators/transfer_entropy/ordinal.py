"""Module for the Ordinal / Permutation transfer entropy estimator."""

from abc import ABC

from numpy import integer, issubdtype, ndarray

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
from ..utils.ordinal import symbolize_series
from ..utils.te_slicing import cte_observations, te_observations


class BaseOrdinalTEEstimator(ABC):
    r"""Base class for the Ordinal / Permutation transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    cond : array-like, optional
        The conditional data used to estimate the conditional transfer entropy.
    embedding_dim : int
        The size of the permutation patterns.
    stable : bool, optional
        If True, when sorting the data, the order of equal elements is preserved.
        This can be useful for reproducibility and testing, but might be slower.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Not compatible with the ``cond`` parameter / conditional TE.
        Alternatively called `offset`.
    step_size : int, optional
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int, optional
        Number of past observations to consider for the source and destination data.
    cond_hist_len : int, optional
        Number of past observations to consider for the conditional data.
        Only used for conditional transfer entropy.

    Raises
    ------
    ValueError
        If the ``embedding_dim`` is negative or not an integer.
    ValueError
        If the ``embedding_dim`` is too large for the given data.
    ValueError
        If ``step_size``, ``prop_time``, and ``embedding_dim`` are such that
        the data is too small.
    TypeError
        If the data are not 1d array-like(s).

    Notes
    -----
    If ``embedding_dim`` is set to 1, the transfer entropy is always 0.
    """

    def __init__(
        self,
        source,
        dest,
        *,  # Enforce keyword-only arguments
        cond=None,
        embedding_dim: int,
        stable: bool = False,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        offset: int = None,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the BaseOrdinalTEEstimator.

        Parameters
        ----------
        source, dest : array-like
            The source (X) and destination (Y) data used to estimate the transfer entropy.
        cond : array-like, optional
            The conditional data used to estimate the conditional transfer entropy.
        embedding_dim : int
            The embedding dimension of the Ordinal entropy.
        stable : bool, optional
            If True, when sorting the data, the order of equal elements is preserved.
            This can be useful for reproducibility and testing, but might be slower.
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
        self.source = None
        self.dest = None
        self.src_hist_len = None
        self.dest_hist_len = None
        self.step_size = None
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
            if self.cond.ndim > 1:
                raise TypeError(
                    "The conditional variable must be an 1d array, "
                    "so that ordinal patterns can be computed from it."
                )
        if not issubdtype(type(embedding_dim), integer) or embedding_dim < 0:
            raise ValueError("The embedding_dim must be a non-negative integer.")
        if embedding_dim == 1:
            logger.warning(
                "The Ordinal mutual information is always 0 for embedding_dim=1."
            )
        if any(var.ndim > 1 for var in (self.source, self.dest)):
            raise TypeError(
                "The data must be tuples of 1D arrays. "
                "Ordinal patterns can only be computed from 1D arrays."
            )
        if not issubdtype(type(step_size), integer) or step_size < 0:
            raise ValueError("The step_size must be a non-negative integer.")
        if len(self.source) < (embedding_dim - 1) * step_size + 1:
            raise ValueError(
                "The data is too small for the given step_size and embedding_dim."
            )
        self.embedding_dim = embedding_dim
        self.stable = stable


class OrdinalTEEstimator(
    BaseOrdinalTEEstimator, PValueMixin, EffectiveValueMixin, TransferEntropyEstimator
):
    r"""Estimator for the Ordinal / Permutation transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    embedding_dim : int
        The size of the permutation patterns.
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

    Raises
    ------
    ValueError
        If the ``embedding_dim`` is negative or not an integer.
    ValueError
        If the ``embedding_dim`` is too large for the given data.
    ValueError
        If ``step_size``, ``prop_time``, and ``embedding_dim`` are such that
        the data is too small.

    Notes
    -----
    If ``embedding_dim`` is set to 1, the transfer entropy is always 0.
    """

    def _calculate(self) -> float:
        """Calculate the Ordinal / Permutation transfer entropy."""
        self.symbols = tuple(
            symbolize_series(
                var, self.embedding_dim, self.step_size, to_int=True, stable=self.stable
            )
            for var in (self.source, self.dest)
        )
        return combined_te_form(
            te_observations,
            *self.symbols,
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
            *self.symbols,
            local=True,
            log_func=self._log_base,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            step_size=self.step_size,
            permute_src=self.permute_src,
            resample_src=self.resample_src,
        )


class OrdinalCTEEstimator(BaseOrdinalTEEstimator, ConditionalTransferEntropyEstimator):
    r"""Estimator for the Ordinal / Permutation conditional transfer entropy.

    Attributes
    ----------
    source, dest, cond : array-like
        The source (X), destination (Y), and conditional (Z) data used to estimate the
        conditional transfer entropy.
    embedding_dim : int
        The size of the permutation patterns.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Assumed time taken by info to transfer from source to destination.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len, cond_hist_len : int
        Number of past observations to consider for the source, destination, and conditional data.

    Raises
    ------
    ValueError
        If the ``embedding_dim`` is negative or not an integer.
    ValueError
        If the ``embedding_dim`` is too large for the given data.
    ValueError
        If ``step_size``, ``prop_time``, and ``embedding_dim`` are such that the
        data is too small.

    Notes
    -----
    If ``embedding_dim`` is set to 1, the transfer entropy is always 0.
    """

    def _calculate(self) -> float:
        """Calculate the Ordinal / Permutation conditional transfer entropy."""
        self.symbols = tuple(
            symbolize_series(
                var, self.embedding_dim, self.step_size, to_int=True, stable=self.stable
            )
            for var in (self.source, self.dest, self.cond)
        )
        return combined_te_form(
            cte_observations,
            *self.symbols,
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
            *self.symbols,
            local=True,
            log_func=self._log_base,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            cond_hist_len=self.cond_hist_len,
            step_size=self.step_size,
        )
