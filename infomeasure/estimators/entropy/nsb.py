"""Module for the NSB (Nemenman-Shafee-Bialek) entropy estimator."""

from functools import lru_cache

from numpy import exp, log, nan
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root_scalar
from scipy.special import loggamma, digamma, polygamma

from infomeasure.estimators.base import DiscreteHEstimator
from ...utils.config import logger
from ...utils.exceptions import TheoreticalInconsistencyError
from ... import Config
from ...utils.types import LogBaseType


@lru_cache(maxsize=1024)
def _cached_polygamma(n, x):
    return polygamma(n, x)


class NsbEntropyEstimator(DiscreteHEstimator):
    r"""NSB (Nemenman-Shafee-Bialek) entropy estimator.

    The NSB estimator provides a Bayesian estimate of Shannon entropy for discrete data
    using the Nemenman, Shafee, Bialek algorithm. This estimator is particularly effective
    for undersampled data where traditional estimators may be biased.

    The NSB estimate is computed as:

    .. math::

        \hat{H}^{\text{NSB}} = \frac{ \int_0^{\ln(K)} d\xi \, \rho(\xi, \textbf{n}) \langle H^m \rangle_{\beta (\xi)}  }
                                    { \int_0^{\ln(K)} d\xi \, \rho(\xi\mid \textbf{n})}

    where

    .. math::

        \rho(\xi \mid \textbf{n}) =
            \mathcal{P}(\beta (\xi)) \frac{ \Gamma(\kappa(\xi))}{\Gamma(N + \kappa(\xi))}
            \prod_{i=1}^K \frac{\Gamma(n_i + \beta(\xi))}{\Gamma(\beta(\xi))}

    The algorithm uses numerical integration to compute the Bayesian posterior over
    possible entropy values, providing a principled approach to entropy estimation
    that accounts for sampling uncertainty :cite:p:`nemenmanEntropyInferenceRevisited2002`.

    If there are no coincidences in the data (all observations are unique), NSB returns NaN
    as the estimator requires repeated observations to function properly.

    Parameters
    ----------
    *data : array-like
        The data used to estimate the entropy.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    base : LogBaseType, default=Config.get("base")
        The logarithm base for entropy calculation.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.

    Notes
    -----
    The NSB estimator is computationally intensive as it requires numerical integration
    and optimisation. For large datasets or when computational efficiency is critical,
    consider using the asymptotic NSB (ANSB) estimator
    :class:`~infomeasure.estimators.entropy.ansb.AnsbEntropyEstimator` instead.

    The estimator assumes a uniform prior over the space of possible probability
    distributions and uses Bayesian inference to estimate the entropy.

    Examples
    --------
    >>> import infomeasure as im
    >>> data = [1, 2, 3, 4, 5, 1, 2]  # Some repeated values
    >>> im.entropy(data, approach='nsb')
    np.float64(1.4526460202102247)
    """

    def __init__(self, *data, K: int = None, base: LogBaseType = Config.get("base")):
        """Initialize the NSB entropy estimator."""
        super().__init__(*data, base=base)
        self.k_given = K

    def _simple_entropy(self):
        """Calculate the NSB entropy of the data.

        Returns
        -------
        float
            The calculated NSB entropy.
        """
        N = self.data[0].N
        K = self.data[0].K if self.k_given is None else self.k_given
        counts = self.data[0].counts
        log_K = log(K)

        # Calculate coincidences (number of repeated observations)
        coincidences = N - K

        if coincidences == 0:
            logger.warning("No coincidences in data - NSB estimator returns NaN")
            return nan

        # Find the extremum of log rho for numerical stability
        l0 = self._find_l0(K, N, counts)

        # Numerical integration for NSB estimate
        try:
            # Numerator: integral of rho * bayes_entropy
            numerator, _ = quad(
                lambda beta: (
                    exp(-self._neg_log_rho(beta, K, N, counts) + l0)
                    * self._dxi(beta, K)
                    * self._bayes(beta, K, counts)
                ),
                0,
                log_K,
                limit=100,
                # epsabs=1e-8,
                # epsrel=1e-6,
            )

            # Denominator: integral of rho
            denominator, _ = quad(
                lambda beta: (
                    exp(-self._neg_log_rho(beta, K, N, counts) + l0)
                    * self._dxi(beta, K)
                ),
                0,
                log_K,
                limit=100,
                # epsabs=1e-8,
                # epsrel=1e-6,
            )

            if denominator == 0:
                logger.warning("NSB integration failed - denominator is zero")
                return nan

            entropy_nats = numerator / denominator

            # Convert to desired base if needed
            if self.base != "e":
                entropy_nats /= log(self.base)

            return entropy_nats

        except Exception as e:
            logger.warning(f"NSB integration failed: {e}")
            return nan

    def _dlogrho(self, K0, K1, N):
        """Calculate derivative of log rho (equation 15 from the paper)."""
        return K1 / K0 - digamma(K0 + N) + digamma(K0)

    def _find_extremum_log_rho(self, K, N):
        """Find the extremum of log rho."""

        def func(K0):
            return self._dlogrho(K0, K, N)

        try:
            result = root_scalar(func, bracket=[0.1, K], method="brentq")
            return result.root
        except ValueError:
            # If bracketing fails, use a simple search
            result = minimize_scalar(
                lambda K0: abs(func(K0)), bounds=(0.1, K), method="bounded"
            )
            return result.x

    def _neg_log_rho(self, beta, K, N, counts):
        """Calculate negative log rho (equation 8 from the paper).

        This implements the negative logarithm of rho(xi|n) from the NSB paper.
        """
        kappa = K * beta

        # Main term: -(loggamma(kappa) - loggamma(N + kappa))
        result = -(loggamma(kappa) - loggamma(N + kappa))

        # Sum over counts: -sum(n_i * (loggamma(n_i + beta) - loggamma(beta)))
        result -= (counts * (loggamma(counts + beta) - loggamma(beta))).sum()

        return result

    def _find_l0(self, K, N, counts):
        """Find l0 for numerical stability."""
        extremum_beta = self._find_extremum_log_rho(K, N) / K
        return self._neg_log_rho(extremum_beta, K, N, counts)

    def _dxi(self, beta, K):
        """Calculate the derivative dxi/dbeta."""
        # The derivative of ξ = ψ(kappa + 1) - ψ(β + 1)
        return K * _cached_polygamma(1, 1 + K * beta) - _cached_polygamma(1, 1 + beta)

    def _bayes(self, beta, K, counts):
        """Calculate the Bayesian entropy expectation ⟨H^m⟩_β(ξ).

        This calculates the expected entropy under a Dirichlet distribution
        with parameters α_i = n_i + β for each bin.

        The expected entropy is:
        E[H] = ψ(Σα_i + 1) - (1/Σα_i) * Σ(α_i * ψ(α_i + 1))
        """
        # Calculate Dirichlet parameters: α_i = n_i + β
        # alphas = counts + beta
        total_alpha = counts.sum() + len(counts) * beta

        # Expected entropy under Dirichlet distribution
        # E[H] = ψ(Σα_i + 1) - (1/Σα_i) * Σ(α_i * ψ(α_i + 1))
        entropy = digamma(total_alpha + 1)

        entropy -= ((counts + beta) / total_alpha * digamma(counts + beta + 1)).sum()

        return entropy

    def _extract_local_values(self):
        """Extract local values for NSB estimator.

        Raises
        ------
        TheoreticalInconsistencyError
            Local values cannot be meaningfully extracted for NSB estimator.
        """
        raise TheoreticalInconsistencyError(
            "Local values extraction is not implemented for NSB estimator. "
            "The NSB estimator is based on Bayesian integration over the entire "
            "probability space and does not provide meaningful local entropy values "
            "for individual data points."
        )

    def _cross_entropy(self):
        """Calculate cross-entropy between two distributions.

        Raises
        ------
        TheoreticalInconsistencyError
            Cross-entropy is not theoretically sound for NSB estimator.
        """
        raise TheoreticalInconsistencyError(
            "Cross-entropy is not implemented for NSB estimator. "
            "The NSB estimator is designed for single distribution entropy estimation "
            "using Bayesian inference and does not extend to cross-entropy "
            "calculations between different distributions."
        )
