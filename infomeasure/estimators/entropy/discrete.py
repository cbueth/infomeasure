"""Module for the discrete entropy estimator."""

from numpy import sum as np_sum, ndarray

from ..base import DiscreteHEstimator
from ...utils.config import logger


class DiscreteEntropyEstimator(DiscreteHEstimator):
    """Estimator for discrete entropy (Shannon entropy).

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
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
