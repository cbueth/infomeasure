"""Module for the Renyi transfer entropy estimator."""

from abc import ABC

from ... import Config
from ...utils.types import LogBaseType
from ..base import (
    EffectiveValueMixin,
    TransferEntropyEstimator,
    ConditionalTransferEntropyEstimator,
)
from ..entropy.renyi import RenyiEntropyEstimator


class BaseRenyiTEEstimator(ABC):
    r"""Base class for the Renyi transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    cond : array-like, optional
        The conditional data used to estimate the conditional transfer entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    alpha : float | int
        The Rényi parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
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

    Raises
    ------
    ValueError
        If the Renyi parameter is not a positive number.
    ValueError
        If the number of nearest neighbors is not a positive integer.
    ValueError
        If the step_size is not a non-negative integer.
    """

    def __init__(
        self,
        source,
        dest,
        cond=None,
        *,  # all following parameters are keyword-only
        k: int = 4,
        alpha: float | int = None,
        noise_level=1e-8,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the BaseRenyiTEEstimator.

        Parameters
        ----------
        source, dest : array-like
            The source (X) and destination (Y) data used to estimate the transfer entropy.
        cond : array-like, optional
            The conditional data used to estimate the conditional transfer entropy.
        k : int
            The number of nearest neighbors to consider.
        alpha : float | int
            The Renyi parameter, order or exponent.
            Sometimes denoted as :math:`\alpha` or :math:`q`.
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
        """
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
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("The Renyi parameter must be a positive number.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        if not isinstance(step_size, int) or step_size < 0:
            raise ValueError("The step_size must be a non-negative integer.")
        self.k = k
        self.alpha = alpha
        self.noise_level = noise_level


class RenyiTEEstimator(
    BaseRenyiTEEstimator, EffectiveValueMixin, TransferEntropyEstimator
):
    r"""Estimator for the Renyi transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    alpha : float | int
        The Rényi parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Assumed time taken by info to transfer from source to destination.
    step_size : int, optional
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int, optional
        Number of past observations to consider for the source and destination data.

    Raises
    ------
    ValueError
        If the Renyi parameter is not a positive number.
    ValueError
        If the number of nearest neighbors is not a positive integer.
    ValueError
        If the step_size is not a non-negative integer.
    """

    def _calculate(self):
        """Estimate the Renyi transfer entropy."""
        return self._generic_te_from_entropy(
            estimator=RenyiEntropyEstimator,
            noise_level=self.noise_level,
            kwargs=dict(k=self.k, alpha=self.alpha, base=self.base),
        )


class RenyiCTEEstimator(BaseRenyiTEEstimator, ConditionalTransferEntropyEstimator):
    r"""Estimator for the Renyi conditional transfer entropy.

    Attributes
    ----------
    source, dest, cond : array-like
        The source (X), destination (Y), and conditional (Z) data used to estimate the
        conditional transfer entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    alpha : float | int
        The Rényi parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.
    noise_level : float
        The standard deviation of the Gaussian noise to add to the data to avoid
        issues with zero distances.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len, cond_hist_len : int
        Number of past observations to consider for the source, destination,
        and conditional data.

    Raises
    ------
    ValueError
        If the Renyi parameter is not a positive number.
    ValueError
        If the number of nearest neighbors is not a positive integer.
    ValueError
        If the step_size is not a non-negative integer.
    """

    def _calculate(self):
        """Estimate the Renyi conditional transfer entropy."""
        return self._generic_cte_from_entropy(
            estimator=RenyiEntropyEstimator,
            noise_level=self.noise_level,
            kwargs=dict(k=self.k, alpha=self.alpha, base=self.base),
        )
