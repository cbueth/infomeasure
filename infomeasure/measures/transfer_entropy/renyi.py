"""Module for the Renyi transfer entropy estimator."""

from ... import Config
from ...utils.types import LogBaseType
from ..base import (
    EffectiveTEMixin,
    TransferEntropyEstimator,
)
from ..entropy.renyi import RenyiEntropyEstimator


class RenyiTEEstimator(EffectiveTEMixin, TransferEntropyEstimator):
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
        If the Renyi parameter is not a positive number.  # TODO: Correct condition?
    ValueError
        If the number of nearest neighbors is not a positive integer.
    ValueError
        If the step_size is not a non-negative integer.
    """

    def __init__(
        self,
        source,
        dest,
        k: int = 4,
        alpha: float | int = None,
        noise_level=1e-8,
        offset: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the Renyi transfer entropy estimator.

        Parameters
        ----------
        k : int
            The number of nearest neighbors to consider.
        alpha : float | int
            The Renyi parameter, order or exponent.
            Sometimes denoted as :math:`\alpha` or :math:`q`.
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

    def _calculate(self):
        """Estimate the Renyi transfer entropy.

        Returns
        -------
        float
            The Renyi transfer entropy.
        """
        # Ensure source and dest are numpy arrays
        self.source = self.source.astype(float)
        self.dest = self.dest.astype(float)

        return self._generic_te_from_entropy(
            estimator=RenyiEntropyEstimator,
            noise_level=self.noise_level,
            kwargs=dict(k=self.k, alpha=self.alpha, base=self.base),
        )
