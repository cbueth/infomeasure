"""Module for the Rényi entropy estimator."""

from numpy import column_stack, issubdtype, integer

from ..base import EntropyEstimator
from ..utils.array import assure_2d_data
from ..utils.exponential_family import (
    calculate_common_entropy_components,
    exponential_family_iq,
    exponential_family_i1,
)
from ... import Config
from ...utils.types import LogBaseType


class RenyiEntropyEstimator(EntropyEstimator):
    r"""Estimator for the Rényi entropy.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    alpha : float | int
        The Rényi parameter, order or exponent.
        Sometimes denoted as :math:`\alpha` or :math:`q`.

    Raises
    ------
    ValueError
        If the Renyi parameter is not a positive number.
    ValueError
        If the number of nearest neighbors is not a positive integer.

    Notes
    -----
    The Rényi entropy is a generalization of Shannon entropy,
    where the small values of probabilities are emphasized for :math:`\alpha < 1`,
    and higher probabilities are emphasized for :math:`\alpha > 1`.
    For :math:`\alpha = 1`, it reduces to Shannon entropy.
    The Rényi-Entropy class can be particularly interesting for systems where additivity
    (in Shannon sense) is not always preserved, especially in nonlinear complex systems,
    such as when dealing with long-range forces.
    """

    def __init__(
        self,
        *data,
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
        super().__init__(*data, base=base)
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("The Renyi parameter must be a positive number.")
        if not issubdtype(type(k), integer) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        self.k = k
        self.alpha = alpha
        self.data = tuple(assure_2d_data(var) for var in self.data)

    def _simple_entropy(self):
        """Calculate the Renyi entropy of the data.

        Returns
        -------
        float
            Renyi entropy of the data.
        """
        V_m, rho_k, N, m = calculate_common_entropy_components(self.data[0], self.k)

        if self.alpha != 1:
            # Renyi entropy for alpha != 1
            I_N_k_a = exponential_family_iq(self.k, self.alpha, V_m, rho_k, N - 1, m)
            if I_N_k_a == 0:
                return 0
            return self._log_base(I_N_k_a) / (1 - self.alpha)
        else:
            # Shannon entropy (limes for alpha = 1)
            return exponential_family_i1(self.k, V_m, rho_k, N - 1, m, self._log_base)

    def _joint_entropy(self):
        """Calculate the joint Renyi entropy of the data.

        This is done by joining the variables into one space
        and calculating the entropy.

        Returns
        -------
        float
            The calculated joint entropy.
        """
        self.data = (column_stack(self.data[0]),)
        return self._simple_entropy()

    def _cross_entropy(self) -> float:
        """Calculate the cross-entropy between two distributions.

        Returns
        -------
        float
            The calculated cross-entropy.
        """
        V_m, rho_k, M, m = calculate_common_entropy_components(
            self.data[1], self.k, at=self.data[0]
        )

        if self.alpha != 1:
            # Renyi cross-entropy for alpha != 1
            I_N_k_a = exponential_family_iq(self.k, self.alpha, V_m, rho_k, M, m)
            if I_N_k_a == 0:
                return 0.0
            return self._log_base(I_N_k_a) / (1 - self.alpha)
        else:
            # Shannon cross-entropy (limes for alpha = 1)
            return exponential_family_i1(self.k, V_m, rho_k, M, m, self._log_base)
