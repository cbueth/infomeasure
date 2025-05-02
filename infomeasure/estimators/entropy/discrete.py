"""Module for the discrete entropy estimator."""

from numpy import sum as np_sum, ndarray, asarray

from ..base import EntropyEstimator, DistributionMixin
from ..utils.ordinal import reduce_joint_space
from ..utils.unique import unique_vals
from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType


class DiscreteEntropyEstimator(DistributionMixin, EntropyEstimator):
    """Estimator for discrete entropy (Shannon entropy).

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    """

    def __init__(self, *data, base: LogBaseType = Config.get("base")):
        """Initialize the DiscreteEntropyEstimator."""
        super().__init__(*data, base=base)
        # warn if the data looks like a float array
        for i_var in range(len(data)):
            if (
                isinstance(self.data[i_var], ndarray)
                and self.data[i_var].dtype.kind == "f"
            ):
                logger.warning(
                    "The data looks like a float array ("
                    f"{self.data[i_var].dtype}). "
                    "Make sure it is properly symbolized or discretized "
                    "for the entropy estimation."
                )
            elif isinstance(self.data[i_var], tuple) and any(
                isinstance(marginal, ndarray) and marginal.dtype.kind == "f"
                for marginal in self.data[i_var]
            ):
                logger.warning(
                    "Some of the data looks like a float array. "
                    "Make sure it is properly symbolized or discretized "
                    "for the entropy estimation."
                )
        # reduce any joint space if applicable
        reduce = tuple(
            (isinstance(var, ndarray) and var.ndim > 1) or isinstance(var, tuple)
            for var in self.data
        )
        if any(reduce):
            # As the discrete shannon entropy disregards the order of the data,
            # we can reduce the values to unique integers.
            # In case of having multiple random variables (tuple or list),
            # this enumerates the unique co-occurrences.
            self.data = tuple(
                reduce_joint_space(var) if red else var
                for var, red in zip(self.data, reduce)
            )

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
