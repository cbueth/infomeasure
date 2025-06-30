"""Module for the discrete Miller-Madow entropy estimator."""

from numpy import log
from numpy import sum as np_sum

from infomeasure.estimators.base import DiscreteHEstimator
from ...utils.config import logger


class MillerMadowEntropyEstimator(DiscreteHEstimator):
    r"""Discrete Miller-Madow entropy estimator.

    .. math::

        \hat{H}_{\tiny{MM}} = \hat{H}_{\tiny{MLE}} + \frac{K - 1}{2N}

    :math:`\hat{H}_{\tiny{MM}}` is the Miller-Madow entropy,
    where :math:`\hat{H}_{\tiny{MLE}}` is the maximum likelihood entropy
    (:class:`~infomeasure.estimators.entropy.discrete.DiscreteEntropyEstimator`).
    :math:`K` is the number of unique values in the data,
    and :math:`N` is the number of observations.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    """

    def _simple_entropy(self):
        """Calculate the Miller-Madow entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        probabilities = self.data[0].probabilities

        correction = self._mm_factor()
        # Calculate the entropy
        return -np_sum(probabilities * self._log_base(probabilities)) + correction

    def _extract_local_values(self):
        """Separately, calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of entropy.
        """
        dist_dict = self.data[0].distribution_dict
        p_local = [dist_dict[val] for val in self.data[0].data]

        correction = self._mm_factor()

        return -self._log_base(p_local) + correction

    def _mm_factor(self):
        # Miller-Madow correction factor
        K = self.data[0].K  # number of unique values
        N = self.data[0].N  # total observations
        correction = (K - 1) / (2 * N)
        if self.base != "e":
            correction /= log(self.base)
        return correction

    def _cross_entropy(self) -> float:
        """Calculate the Miller-Madow cross-entropy between two distributions.

        Returns
        -------
        float
            The calculated cross-entropy.
        """
        # Calculate the distribution of both data sets
        dist_p = self.data[0].distribution_dict
        uniq_p = self.data[0].uniq
        dist_q = self.data[1].distribution_dict
        uniq_q = self.data[1].uniq
        # Only consider the values where both RV have the same support
        uniq = list(set(uniq_p).intersection(set(uniq_q)))  # P ∩ Q
        if len(uniq) == 0:
            logger.warning("No common support between the two distributions.")
            return 0.0
        # Miller-Madow correction
        N = self.data[0].N + self.data[1].N
        K = ((self.data[0].K + self.data[1].K) / 2.0) - 1.0
        correction = K / N if self.base == "e" else K / (N * log(self.base))
        return (
            -np_sum([dist_p[val] * self._log_base(dist_q[val]) for val in uniq])
            + correction
        )
