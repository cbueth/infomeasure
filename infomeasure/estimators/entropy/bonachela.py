"""Module for the Bonachela entropy estimator."""

from numpy import log
from numpy import arange, sum as np_sum

from infomeasure.estimators.base import DiscreteHEstimator
from infomeasure.utils.exceptions import TheoreticalInconsistencyError


class BonachelaEntropyEstimator(DiscreteHEstimator):
    r"""Bonachela (Bonachela-Hinrichsen-Muñoz) entropy estimator for discrete data.

    The Bonachela estimator computes the Shannon entropy using the formula from
    :cite:p:`bonachelaEntropyEstimatesSmall2008`:

    .. math::

        \hat{H}_{B} = \frac{1}{N+2} \sum_{i=1}^{K} \left( (n_i + 1) \sum_{j=n_i + 2}^{N+2} \frac{1}{j} \right)

    where :math:`n_i` are the counts for each unique value, :math:`K` is the number of
    unique values, and :math:`N` is the total number of observations.

    This estimator is specially designed to provide a compromise between low bias and
    small statistical errors for short data series, particularly when the data sets are
    small and the probabilities are not close to zero.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    """

    def _simple_entropy(self):
        """Calculate the Bonachela entropy of the data.

        Returns
        -------
        float
            The calculated Bonachela entropy.
        """
        # Get counts and total observations
        counts = self.data[0].counts
        N = self.data[0].N

        # Vectorized computation
        # For each count ni = count + 1, we need sum(1/j for j in range(ni + 1, N + 3))
        ni_values = counts + 1  # Shape: (K,)

        # Create array of all possible j values from 2 to N+2
        j_values = arange(2, N + 3)  # Shape: (N+1,)

        # Create a mask matrix where mask[i, j] is True if j_values[j] > ni_values[i]
        # This uses broadcasting: ni_values[:, None] has shape (K, 1), j_values has shape (N+1,)
        mask = j_values[None, :] > ni_values[:, None]  # Shape: (K, N+1)

        # Create reciprocal array: 1/j for each j
        reciprocals = 1.0 / j_values  # Shape: (N+1,)

        # Apply mask and sum along j dimension to get inner sums for each count
        # mask * reciprocals[None, :] broadcasts reciprocals to shape (K, N+1)
        inner_sums = np_sum(mask * reciprocals[None, :], axis=1)  # Shape: (K,)

        # Calculate contributions: ni * inner_sum for each count
        contributions = ni_values * inner_sums  # Shape: (K,)

        # Sum all contributions
        acc = np_sum(contributions)

        # Calculate final entropy with normalization factor
        ent = acc / (N + 2)

        # Convert to the desired base if needed
        if self.base != "e":
            ent /= log(self.base)

        return ent

    def _extract_local_values(self):
        """Calculate local Bonachela entropy values for each data point.

        Returns
        -------
        ndarray[float]
            The calculated local values of Bonachela entropy.
        """
        raise TheoreticalInconsistencyError(
            "Local values are not implemented for Bonachela estimator due to "
            "theoretical inconsistencies in the mathematical foundation."
        )

    def _cross_entropy(self):
        """Calculate cross-entropy between two distributions using Bonachela estimator.

        Returns
        -------
        float
            The calculated cross-entropy.
        """
        raise TheoreticalInconsistencyError(
            "Cross-entropy is not implemented for Bonachela estimator due to "
            "theoretical inconsistencies in applying bias corrections from "
            "different distributions."
        )
