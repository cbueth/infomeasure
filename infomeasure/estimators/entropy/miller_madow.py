"""Module for the discrete Miller-Madow entropy estimator."""

from numpy import asarray, log
from numpy import sum as np_sum

from infomeasure.estimators.base import (
    DistributionMixin,
    EntropyEstimator,
    DiscreteMixin,
)
from ..utils.unique import unique_vals
from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType


class MillerMadowEntropyEstimator(DistributionMixin, DiscreteMixin, EntropyEstimator):
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

    def __init__(self, *data, base: LogBaseType = Config.get("base")):
        """Initialize the MillerMadowEntropyEstimator."""
        super().__init__(*data, base=base)
        # warn if the data looks like a float array
        self._check_data()
        # reduce any joint space if applicable
        self._reduce_space()

    def _simple_entropy(self):
        """Calculate the Miller-Madow entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        uniq, counts, self.dist_dict = unique_vals(self.data[0])
        probabilities = asarray(list(self.dist_dict.values()))

        # Miller-Madow correction factor
        K = len(self.dist_dict)  # number of unique values
        N = len(self.data[0])  # total observations
        correction = (K - 1) / (2 * N)
        if self.base != "e":
            correction = correction / log(self.base)
        # Calculate the entropy
        return -np_sum(probabilities * self._log_base(probabilities)) + correction

    def _joint_entropy(self):
        """Calculate the joint Miller-Madow entropy of the data.

        Returns
        -------
        float
            The calculated joint entropy.
        """
        # The data has already been reduced to unique values of co-occurrences
        return self._simple_entropy()

    def _cross_entropy(self) -> float:
        """Calculate the Miller-Madow cross-entropy between two distributions.

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
        # Miller-Madow correction
        N = len(uniq_p) + len(uniq_q)
        K = ((len(self.data[0]) + len(self.data[1])) / 2.0) - 1.0
        correction = K / N if self.base == "e" else K / (N * log(self.base))
        return (
            -np_sum([dist_p[val] * self._log_base(dist_q[val]) for val in uniq])
            + correction
        )
