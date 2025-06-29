"""Module for the Bayesian entropy estimator."""

from numpy import sum as np_sum

from infomeasure.estimators.base import DiscreteHEstimator
from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType


class BayesEntropyEstimator(DiscreteHEstimator):
    r"""Bayesian entropy estimator.

    Computes an estimate of Shannon entropy using Bayesian probability estimates with
    a Dirichlet prior characterized by concentration parameter α. This approach provides
    a principled way to handle sparse data and incorporate prior knowledge about the
    probability distribution.

    The Bayesian probabilities are calculated as:

    .. math::

        p_k^{\text{Bayes}} = \frac{n_k + \alpha}{N + K \alpha}

    where :math:`n_k` is the count of symbol :math:`k`, :math:`N` is the total number
    of observations, :math:`K` is the support size (number of unique symbols), and
    :math:`\alpha` is the concentration parameter of the Dirichlet prior.

    The entropy is then :math:`-\sum p_k^{\text{Bayes}} \log p_k^{\text{Bayes}}`,
    same as the maximum likelihood entropy estimator,
    also supporting local entropy values.

    **Concentration Parameter Choices**

    The concentration parameter α controls the strength of the prior belief in uniform
    distribution. Several well-established choices are available:

    **Jeffreys Prior** (``α = 0.5 = "jeffrey"``)
        Non-informative prior that is invariant under reparameterization.
        Provides good performance for most applications
        :cite:p:`krichevskyPerformanceUniversalEncoding1981`.

    **Laplace Prior** (``α = 1.0 = "laplace"``)
        Uniform prior that adds one pseudocount to each symbol
        :cite:p:`bayesEssaySolvingProblem1763`.
        Simple and widely used, equivalent to add-one smoothing.

    **Schürmann-Grassberger Prior** (``α = 1/K = "sch-grass"``)
        Adaptive prior that scales with the alphabet size.
        Particularly effective for large alphabets.

    **Minimax Prior** (``α = √N/K = "min-max"``)
        Minimises the maximum expected loss.
        Balances between sample size and alphabet size.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    alpha : float
        The concentration parameter α of the Dirichlet prior.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    """

    def __init__(
        self,
        *data,
        alpha: float | str,
        K: int = None,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the BayesEntropyEstimator.

        Parameters
        ----------
        *data : array-like
            The data used to estimate the entropy.
        alpha : float | str
            The concentration parameter α.
            Either a float or a string specifying the choice of concentration parameter.
        K : int, optional
            The support size. If not provided, uses the observed support size.
        base : LogBaseType, default=Config.get("base")
            The logarithm base for entropy calculation.
        """
        super().__init__(*data, base=base)

        self.alpha = alpha
        self.K_param = K

    def _simple_entropy(self):
        """Calculate the Bayesian entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        bayes_probs = self.bayes_probs

        # Calculate entropy: -sum(p_k * log(p_k))
        return -np_sum(bayes_probs * self._log_base(bayes_probs))

    @property
    def bayes_probs(self):
        K = self.K_param if self.K_param is not None else self.data[0].K
        N = self.data[0].N
        self._set_alpha(K, N)
        # Calculate Bayesian probabilities: p_k = (n_k + α) / (N + K*α)
        weight = N + K * self.alpha
        bayes_probs = (self.data[0].counts + self.alpha) / weight
        return bayes_probs

    @property
    def dist_dict(self):
        """Return the Bayesian distribution dictionary for JSD."""
        return dict(zip(self.data[0].uniq, self.bayes_probs))

    def _set_alpha(self, K, N):
        # Alpha
        if isinstance(self.alpha, (int, float)):
            pass
        elif not isinstance(self.alpha, str) or (
            isinstance(self.alpha, str)
            and self.alpha.lower() not in ["jeffrey", "laplace", "sch-grass", "min-max"]
        ):
            raise ValueError(
                "Concentration parameter must be a float or one of the following"
                "strings: \n'jeffrey', 'laplace', 'sch-grass', 'min-max'\n"
                f"Received: {self.alpha}"
            )
        elif self.alpha.lower() == "jeffrey":
            self.alpha = 0.5
        elif self.alpha.lower() == "laplace":
            self.alpha = 1.0
        elif self.alpha.lower() == "sch-grass":
            self.alpha = 1 / K
        else:  # self.alpha == "min-max":
            self.alpha = N**0.5 / K

    def _cross_entropy(self) -> float:
        """Calculate the Bayesian cross-entropy between two distributions.

        Returns
        -------
        float
            The calculated cross-entropy.
        """

        K_p = self.K_param if self.K_param is not None else self.data[0].K
        K_q = self.K_param if self.K_param is not None else self.data[1].K
        N_p = self.data[0].N
        N_q = self.data[1].N

        # Calculate Bayesian distributions using consistent weight calculation
        weight_p = N_p + K_p * self.alpha
        weight_q = N_q + K_q * self.alpha

        dist_p = {}
        for val, count in zip(self.data[0].uniq, self.data[0].counts):
            dist_p[val] = (count + self.alpha) / weight_p

        dist_q = {}
        for val, count in zip(self.data[1].uniq, self.data[1].counts):
            dist_q[val] = (count + self.alpha) / weight_q

        # Find common support
        uniq_p = set(self.data[0].uniq)
        uniq_q = set(self.data[1].uniq)
        uniq = list(uniq_p.intersection(uniq_q))

        if len(uniq) == 0:
            logger.warning("No common support between the two distributions.")
            return 0.0

        # Calculate cross-entropy: -sum(p(x) * log(q(x)))
        return -np_sum([dist_p[val] * self._log_base(dist_q[val]) for val in uniq])
