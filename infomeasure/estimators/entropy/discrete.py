"""Module for the discrete entropy estimator."""

from numpy import sum as np_sum, ndarray, asarray

from ..base import EntropyEstimator, DistributionMixin, DiscreteMixin
from ..utils.unique import unique_vals
from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType


class DiscreteEntropyEstimator(DiscreteMixin, DistributionMixin, EntropyEstimator):
    """Estimator for discrete entropy (Shannon entropy).

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    """

    def __init__(self, *data, base: LogBaseType = Config.get("base")):
        """Initialise the DiscreteEntropyEstimator."""
        super().__init__(*data, base=base)
        # warn if the data looks like a float array
        self._check_data()
        # reduce any joint space if applicable
        self._reduce_space()

    def _simple_entropy(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        uniq, counts, self.dist_dict = unique_vals(self.data[0])
        probabilities = asarray(list(self.dist_dict.values()))
        # Calculate the entropy
        return -np_sum(probabilities * self._log_base(probabilities))

    def _joint_entropy(self):
        """Calculate the joint entropy of the data.

        Returns
        -------
        float
            The calculated joint entropy.
        """
        # The data has already been reduced to unique values of co-occurrences
        return self._simple_entropy()

    def _extract_local_values(self):
        """Separately, calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of entropy.
        """
        p_local = [self.dist_dict[val] for val in self.data[0]]
        return -self._log_base(p_local)

    def _cross_entropy(self) -> float:
        """Calculate the cross-entropy between two distributions.

        Returns
        -------
        float
            The calculated cross-entropy.
        """
        # Calculate distribution of both data sets
        uniq_p, counts_p, dist_p = unique_vals(self.data[0])
        uniq_q, counts_q, dist_q = unique_vals(self.data[1])
        # Only consider the values where both RV have the same support
        uniq = list(set(uniq_p).intersection(set(uniq_q)))  # P ∩ Q
        if len(uniq) == 0:
            logger.warning("No common support between the two distributions.")
            return 0.0
        return -np_sum([dist_p[val] * self._log_base(dist_q[val]) for val in uniq])
