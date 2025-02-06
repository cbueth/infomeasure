"""Module for the Rényi entropy estimator."""

from numpy import column_stack, ndarray, newaxis

from ... import Config
from ...utils.types import LogBaseType
from ..base import EntropyEstimator, PValueMixin
from ..utils.exponential_family import (
    calculate_common_entropy_components,
    exponential_family_iq,
    exponential_family_i1,
)


class RenyiEntropyEstimator(PValueMixin, EntropyEstimator):
    r"""Estimator for the Rényi entropy.

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    alpha : float | int
        The Rényi parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.
    base : int | float | "e", optional
        The logarithm base for the entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.

    Raises
    ------
    ValueError
        If the Renyi parameter is not a positive number.
    ValueError
        If the number of nearest neighbors is not a positive integer.
    """

    def __init__(
        self,
        data,
        *,  # all following parameters are keyword-only
        k: int = 4,
        alpha: float | int = None,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the RenyiEntropyEstimator.

        Parameters
        ----------
        k : int
            The number of nearest neighbors to consider.
        alpha : float | int
            The Renyi parameter, order or exponent.
            Sometimes denoted as :math:`\alpha` or :math:`q`.
        """
        super().__init__(data, base)
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("The Renyi parameter must be a positive number.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        self.k = k
        self.alpha = alpha
        if isinstance(self.data, ndarray) and self.data.ndim == 1:
            self.data = self.data[:, newaxis]
        elif isinstance(self.data, tuple):
            self.data = tuple(
                marginal[:, newaxis]
                if isinstance(marginal, ndarray) and marginal.ndim == 1
                else marginal
                for marginal in self.data
            )

    def _simple_entropy(self):
        """Calculate the Renyi entropy of the data.

        Returns
        -------
        float
            Renyi entropy of the data.
        """
        V_m, rho_k, N, m = calculate_common_entropy_components(self.data, self.k)

        if self.alpha != 1:
            # Renyi entropy for alpha != 1
            I_N_k_a = exponential_family_iq(self.k, self.alpha, V_m, rho_k, N, m)
            return self._log_base(I_N_k_a) / (1 - self.alpha)
        else:
            # Shannon entropy (limes for alpha = 1)
            return exponential_family_i1(self.k, V_m, rho_k, N, m, self._log_base)

    def _joint_entropy(self):
        """Calculate the joint Renyi entropy of the data.

        This is done by joining the variables into one space
        and calculating the entropy.

        Returns
        -------
        float
            The calculated joint entropy.
        """
        self.data = column_stack(self.data)
        return self._simple_entropy()
