"""Module for the discrete entropy estimator."""

from numpy import sum as np_sum, ndarray

from ..base import EntropyEstimator, DistributionMixin
from ..utils.ordinal import reduce_joint_space
from ..utils.unique import histogram_unique_values
from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType


class DiscreteEntropyEstimator(DistributionMixin, EntropyEstimator):
    """Estimator for discrete entropy (Shannon entropy).

    Attributes
    ----------
    data : array-like
        The data used to estimate the entropy.
    """

    def __init__(self, data, *, base: LogBaseType = Config.get("base")):
        """Initialize the DiscreteEntropyEstimator."""
        super().__init__(data, base=base)
        # warn if the data looks like a float array
        if isinstance(self.data, ndarray) and self.data.dtype.kind == "f":
            logger.warning(
                "The data looks like a float array ("
                f"{data.dtype}). "
                "Make sure it is properly symbolized or discretized "
                "for the entropy estimation."
            )
        elif isinstance(self.data, tuple) and any(
            isinstance(marginal, ndarray) and marginal.dtype.kind == "f"
            for marginal in self.data
        ):
            logger.warning(
                "Some of the data looks like a float array. "
                "Make sure it is properly symbolized or discretized "
                "for the entropy estimation."
            )
        if (isinstance(self.data, ndarray) and self.data.ndim > 1) or isinstance(
            self.data, tuple
        ):
            # As the discrete shannon entropy disregards the order of the data,
            # we can reduce the values to unique integers.
            # In case of having multiple random variables (tuple or list),
            # this enumerates the unique co-occurrences.
            self.data = reduce_joint_space(self.data)

    def _simple_entropy(self):
        """Calculate the entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        histogram, self.dist_dict = histogram_unique_values(self.data)
        # Calculate the entropy
        return -np_sum(histogram * self._log_base(histogram))

    def _joint_entropy(self):
        """Calculate the joint entropy of the data.

        Returns
        -------
        float
            The calculated joint entropy.
        """
        # The data has already been reduced to unique values of co-occurrences
        return self._simple_entropy()
