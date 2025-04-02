"""Module for the Tsallis transfer entropy estimator."""

from abc import ABC

from numpy import issubdtype, integer

from ..base import (
    PValueMixin,
    EffectiveValueMixin,
    TransferEntropyEstimator,
    ConditionalTransferEntropyEstimator,
)
from ..entropy.tsallis import TsallisEntropyEstimator
from ... import Config
from ...utils.types import LogBaseType


class BaseTsallisTEEstimator(ABC):
    r"""Base class for the Tsallis transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    cond : array-like, optional
        The conditional data used to estimate the conditional transfer entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    q : float | int
        The Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`q`,
        analogous to the Rényi parameter :math:`\alpha`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
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

    Raises
    ------
    ValueError
        If the Tsallis parameter is not a positive number.
    ValueError
        If the number of nearest neighbors is not a positive integer.
    ValueError
        If the step_size is not a non-negative integer.
    """

    def __init__(
        self,
        source,
        dest,
        *,  # Enforce keyword-only arguments
        cond=None,
        k: int = 4,
        q: float | int = None,
        noise_level=1e-8,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        offset: int = None,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the BaseTsallisTEEstimator.

        Parameters
        ----------
        source, dest : array-like
            The source (X) and destination (Y) data used to estimate the transfer entropy.
        cond : array-like, optional
            The conditional data used to estimate the conditional transfer entropy.
        k : int
            The number of nearest neighbors to consider.
        q : float | int
            The Tsallis parameter, order or exponent.
            Sometimes denoted as :math:`q`,
            analogous to the Rényi parameter :math:`\alpha`.
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
        if not isinstance(q, (int, float)) or q <= 0:
            raise ValueError("The Tsallis parameter ``q`` must be a positive number.")
        if not issubdtype(type(k), integer) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        if not issubdtype(type(step_size), integer) or step_size < 0:
            raise ValueError("The step_size must be a non-negative integer.")
        self.k = k
        self.q = q
        self.noise_level = noise_level


class TsallisTEEstimator(
    BaseTsallisTEEstimator, PValueMixin, EffectiveValueMixin, TransferEntropyEstimator
):
    r"""Estimator for the Tsallis transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    q : float | int
        The Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`q`,
        analogous to the Rényi parameter :math:`\alpha`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Assumed time taken by info to transfer from source to destination.
        Alternatively called `offset`.
    step_size : int, optional
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int, optional
        Number of past observations to consider for the source and destination data.

    Raises
    ------
    ValueError
        If the Tsallis parameter is not a positive number.
    ValueError
        If the number of nearest neighbors is not a positive integer.
    ValueError
        If the step_size is not a non-negative integer.

    Notes
    -----
    In the $q \to 1$ limit, the Jackson sum (q-additivity) reduces to
    ordinary summation,
    and the Tallis entropy reduces to Shannon Entropy.
    This class of entropy measure is in particularly useful in the study in connection
    with long-range correlated systems and with non-equilibrium phenomena.
    """

    def _calculate(self):
        """Estimate the Tsallis transfer entropy."""
        return self._generic_te_from_entropy(
            estimator=TsallisEntropyEstimator,
            noise_level=self.noise_level,
            kwargs=dict(k=self.k, q=self.q, base=self.base),
        )


class TsallisCTEEstimator(BaseTsallisTEEstimator, ConditionalTransferEntropyEstimator):
    r"""Estimator for the Tsallis conditional transfer entropy.

    Attributes
    ----------
    source, dest, cond : array-like
        The source (X), destination (Y) and conditional (Z) data used to estimate the
        conditional transfer entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    q : float | int
        The Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`q`,
        analogous to the Rényi parameter :math:`\alpha`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    step_size : int, optional
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len, cond_hist_len : int, optional
        Number of past observations to consider for the source, destination and conditional
        data.

    Raises
    ------
    ValueError
        If the Tsallis parameter is not a positive number.
    ValueError
        If the number of nearest neighbors is not a positive integer.
    ValueError
        If the step_size is not a non-negative integer.

    Notes
    -----
    In the $q \to 1$ limit, the Jackson sum (q-additivity) reduces to
    ordinary summation,
    and the Tallis entropy reduces to Shannon Entropy.
    This class of entropy measure is in particularly useful in the study in connection
    with long-range correlated systems and with non-equilibrium phenomena.
    """

    def _calculate(self):
        """Estimate the conditional Tsallis transfer entropy."""
        return self._generic_cte_from_entropy(
            estimator=TsallisEntropyEstimator,
            noise_level=self.noise_level,
            kwargs=dict(k=self.k, q=self.q, base=self.base),
        )
