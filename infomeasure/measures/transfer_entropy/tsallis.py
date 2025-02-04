"""Module for the Tsallis transfer entropy estimator."""

from ... import Config
from ...utils.types import LogBaseType
from ..base import (
    EffectiveValueMixin,
    TransferEntropyEstimator,
)
from ..entropy.tsallis import TsallisEntropyEstimator


class TsallisTEEstimator(EffectiveValueMixin, TransferEntropyEstimator):
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
        k: int = 4,
        q: float | int = None,
        noise_level=1e-8,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the Tsallis transfer entropy estimator.

        Parameters
        ----------
        k : int
            The number of nearest neighbors to consider.
        q : float | int
            The Tsallis parameter, order or exponent.
            Sometimes denoted as :math:`q`,
            analogous to the Rényi parameter :math:`\alpha`.
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
        if not isinstance(q, (int, float)) or q <= 0:
            raise ValueError("The Tsallis parameter ``q`` must be a positive number.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        if not isinstance(step_size, int) or step_size < 0:
            raise ValueError("The step_size must be a non-negative integer.")
        self.k = k
        self.q = q
        self.noise_level = noise_level

    def _calculate(self):
        """Estimate the Tsallis transfer entropy.

        Returns
        -------
        float
            The Tsallis transfer entropy.
        """
        return self._generic_te_from_entropy(
            estimator=TsallisEntropyEstimator,
            noise_level=self.noise_level,
            kwargs=dict(k=self.k, q=self.q, base=self.base),
        )
