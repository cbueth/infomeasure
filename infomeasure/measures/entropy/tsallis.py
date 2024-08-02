"""Module for Tsallis entropy estimator."""

from ... import Config
from ...utils.types import LogBaseType
from ..base import EntropyEstimator, PValueMixin
from ..utils.exponential_family import (
    calculate_common_entropy_components,
    exponential_family_iq,
    exponential_family_i1,
)


class TsallisEntropyEstimator(PValueMixin, EntropyEstimator):
    r"""Estimator for the Tsallis entropy.

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    q : float
        The Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`q`, analogous to the Rényi parameter :math:`\alpha`.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Raises
    ------
    ValueError
        If the Renyi parameter is not a positive number.  # TODO: Correct condition?
    ValueError
        If the number of nearest neighbors is not a positive integer.
    """

    def __init__(
        self,
        data,
        k: int = 4,
        q: float | int = None,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the TsallisEntropyEstimator.

        Parameters
        ----------
        k : int
            The number of nearest neighbors to consider.
        q : float | int
            The Tsallis parameter, order or exponent.
            Sometimes denoted as :math:`q`, analogous to the Rényi parameter :math:`\alpha`.
        """
        super().__init__(data, base)
        self.data = self.data.reshape(-1, 1) if self.data.ndim == 1 else self.data
        if not isinstance(q, (int, float)) or q <= 0:
            raise ValueError("The Tsallis parameter ``q`` must be a positive number.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        self.k = k
        self.q = q

    def _calculate(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The Tsallis entropy.
        """
        V_m, rho_k, N, m = calculate_common_entropy_components(self.data, self.k)

        if self.q != 1:
            # Tsallis entropy for q != 1
            I_N_k_q = exponential_family_iq(self.k, self.q, V_m, rho_k, N, m)
            return (1 - I_N_k_q) / (self.q - 1)
        else:
            # Shannon entropy (limes for alpha = 1)
            return exponential_family_i1(self.k, V_m, rho_k, N, m, self._log_base)
