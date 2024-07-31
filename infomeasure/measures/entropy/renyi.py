"""Module for the Rényi entropy estimator."""

from numpy import mean as np_mean, pi
from scipy.special import gamma, digamma
from scipy.spatial import KDTree

from ... import Config
from ...utils.types import LogBaseType
from ..base import EntropyEstimator, PValueMixin


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
        If the Renyi parameter is not a positive number.  # TODO: Correct condition?
    ValueError
        If the number of nearest neighbors is not a positive integer.
    """

    def __init__(
        self,
        data,
        k: int = 4,
        alpha: float | int = None,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the RenyiEntropyEstimator.

        Parameters
        ----------
        k : int
            The number of nearest neighbors to consider.
        alpha : float | int
            The Renyi parameter, order or exponent.
            Sometimes denoted as :math:`\alpha` or :math:`q`.
        """
        super().__init__(data, base)
        self.data = self.data.reshape(-1, 1) if self.data.ndim == 1 else self.data
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("The Renyi parameter must be a positive number.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                "The number of nearest neighbors must be a positive integer."
            )
        self.k = k
        self.alpha = alpha

    def _calculate(self):
        """Calculate the Renyi entropy of the data.

        Returns
        -------
        float
            Renyi entropy of the data.
        """
        N, m = self.data.shape

        # Volume of the unit ball in m-dimensional space
        V_m = pi ** (m / 2) / gamma(m / 2 + 1)

        # Build k-d tree for nearest neighbor search
        tree = KDTree(self.data)

        # Get the k-th nearest neighbor distances
        rho_k = tree.query(self.data, k=self.k + 1)[0][
            :, self.k
        ]  # k+1 because the point itself is included

        if self.alpha != 1:
            C_k = (gamma(self.k) / gamma(self.k + 1 - self.alpha)) ** (
                1 / (1 - self.alpha)
            )
            # Renyi entropy for alpha != 1
            zeta_N_i_k = (N - 1) * C_k * V_m * rho_k**m
            I_N_k_a = np_mean(zeta_N_i_k ** (1 - self.alpha))
            H_N_k_a_star = self._log_base(I_N_k_a) / (1 - self.alpha)
        else:
            # Shannon entropy (limes for alpha = 1)
            psi_k = digamma(self.k)
            zeta_N_i_k = (N - 1) * self._log_base(-psi_k) * V_m * rho_k**m
            I_N_k_a = np_mean(self._log_base(zeta_N_i_k))
            H_N_k_a_star = I_N_k_a  # For Shannon entropy, it's just I_N_k_a

        return H_N_k_a_star
