"""Module for the discrete entropy estimator."""

from numpy import sum as np_sum, ndarray

from ..base import DiscreteHEstimator
from ...utils.config import logger


class DiscreteEntropyEstimator(DiscreteHEstimator):
    r"""Standard Shannon entropy estimator for discrete data using maximum likelihood.

    The discrete entropy estimator computes the Shannon entropy using the classical
    maximum likelihood approach:

    .. math::

        \hat{H} = -\sum_{i=1}^{K} \hat{p}_i \log \hat{p}_i

    where :math:`\hat{p}_i = \frac{n_i}{N}` are the empirical probabilities,
    :math:`n_i` are the counts for each unique value :math:`i`, :math:`K` is the number of
    unique values, and :math:`N` is the total number of observations.

    This is the most fundamental entropy estimator and serves as the baseline for
    comparison with other bias-corrected estimators. While it provides an asymptotically
    unbiased estimate of the true entropy, it can exhibit significant bias for small
    sample sizes, particularly when the number of unique values is large relative to
    the sample size.

    The estimator is suitable for:

    - Large datasets where bias is minimal
    - Baseline comparisons with bias-corrected estimators
    - Applications where computational simplicity is preferred
    - Well-sampled distributions with sufficient observations per unique value

    For small sample sizes or distributions with many rare events, consider using
    bias-corrected estimators
    such as :class:`~infomeasure.estimators.entropy.chao_shen.ChaoShenEntropyEstimator`,
    :class:`~infomeasure.estimators.entropy.bonachela.BonachelaEntropyEstimator`,
    or :class:`~infomeasure.estimators.entropy.zhang.ZhangEntropyEstimator`.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy. For joint entropy, multiple arrays
        can be provided.
    base : float or str, default=Config.get("base")
        The logarithm base for entropy calculation. Common values are 2 (bits),
        10 (dits), or 'e' (nats).

    Examples
    --------
    >>> import infomeasure as im
    >>> # Simple entropy calculation
    >>> data = [1, 1, 2, 3, 3, 4, 5]
    >>> entropy_value = im.entropy(data, approach="discrete")
    >>> print(f"Entropy: {entropy_value:.3f} nats")
    Entropy: 1.550 nats
    >>> # Local values
    >>> estimator = im.estimator(data, measure="h", approach="discrete")
    >>> estimator.local_vals()
    array([1.25276297, 1.25276297, 1.94591015, 1.25276297, 1.25276297,
       1.94591015, 1.94591015])
    """

    def _simple_entropy(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        probabilities = self.data[0].probabilities
        # Calculate the entropy
        return -np_sum(probabilities * self._log_base(probabilities))

    def _extract_local_values(self):
        """Separately, calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of entropy.
        """
        distribution_dict = dict(zip(self.data[0].uniq, self.data[0].probabilities))
        p_local = [distribution_dict[val] for val in self.data[0].data]
        return -self._log_base(p_local)

    def _cross_entropy(self) -> float:
        """Calculate the cross-entropy between two distributions.

        Returns
        -------
        float
            The calculated cross-entropy.
        """
        # Calculate distribution of both data sets
        uniq_p = self.data[0].uniq
        dist_p = self.data[0].distribution_dict
        uniq_q = self.data[1].uniq
        dist_q = self.data[1].distribution_dict
        # Only consider the values where both RV have the same support
        uniq = list(set(uniq_p).intersection(set(uniq_q)))  # P ∩ Q
        if len(uniq) == 0:
            logger.warning("No common support between the two distributions.")
            return 0.0
        return -np_sum([dist_p[val] * self._log_base(dist_q[val]) for val in uniq])
