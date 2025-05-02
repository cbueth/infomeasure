"""Module for Tsallis entropy estimator."""

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


class TsallisEntropyEstimator(EntropyEstimator):
    r"""Estimator for the Tsallis entropy.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    k : int
        The number of nearest neighbors used in the estimation.
    q : float
        The Tsallis parameter, order or exponent.
        Sometimes denoted as :math:`q`, analogous to the Rényi parameter :math:`\alpha`.

    Raises
    ------
    ValueError
        If the Tsallis parameter is not a positive number.
    ValueError
        If the number of nearest neighbors is not a positive integer.

    Notes
    -----
    In the :math:`q \to 1` limit, the Jackson sum (q-additivity) reduces to
    ordinary summation,
    and the Tallis entropy reduces to Shannon Entropy.
    This class of entropy measure is in particularly useful in the study in connection
    with long-range correlated systems and with non-equilibrium phenomena.
    """

    def __init__(
        self,
        *data,
        k: int = 4,
        q: float | int = None,
        base: LogBaseType = Config.get("base"),
    ):
        r"""Initialize the TsallisMIEstimator.

        Parameters
        ----------
        k : int
            The number of nearest neighbors to consider.
        q : float | int
            The Tsallis parameter, order or exponent.
            Sometimes denoted as :math:`q`, analogous to the Rényi parameter :math:`\alpha`.
        """
        super().__init__(*data, base=base)
        if not isinstance(q, (int, float)) or q <= 0:
            raise ValueError("The Tsallis parameter ``q`` must be a positive number.")
        if not issubdtype(type(k), integer) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        self.k = k
        self.q = q
        self.data = tuple(assure_2d_data(var) for var in self.data)

    def _simple_entropy(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The Tsallis entropy.
        """
        V_m, rho_k, N, m = calculate_common_entropy_components(self.data[0], self.k)

        if self.q != 1:
            # Tsallis entropy for q != 1
            I_N_k_q = exponential_family_iq(self.k, self.q, V_m, rho_k, N - 1, m)
            if I_N_k_q == 0:
                return 0.0
            return (1 - I_N_k_q) / (self.q - 1)
        else:
            # Shannon entropy (limes for alpha = 1)
            return exponential_family_i1(self.k, V_m, rho_k, N - 1, m, self._log_base)

    def _joint_entropy(self):
        """Calculate the joint Tsallis entropy of the data.

        This is done by joining the variables into one space
        and calculating the entropy.

        Returns
        -------
        float
            The joint Tsallis entropy.
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

        if self.q != 1:
            # Renyi cross-entropy for q != 1
            I_N_k_q = exponential_family_iq(self.k, self.q, V_m, rho_k, M, m)
            if I_N_k_q == 0:
                return 0.0
            return (1 - I_N_k_q) / (self.q - 1)
        else:
            # Shannon cross-entropy (limes for q = 1)
            return exponential_family_i1(self.k, V_m, rho_k, M, m, self._log_base)
