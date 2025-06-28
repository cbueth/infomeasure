"""Module for the Bonachela entropy estimator."""

from numpy import log

from infomeasure.estimators.base import DiscreteHEstimator


class BonachelaEntropyEstimator(DiscreteHEstimator):
    r"""Bonachela entropy estimator for discrete data.

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
        N = int(self.data[0].N)

        acc = 0.0

        # Iterate over each unique value and its count
        for count in counts:
            # Calculate the inner sum
            t = 0.0
            ni = int(count) + 1

            for j in range(ni + 1, N + 3):  # j from ni+1 to N+2 (inclusive)
                t += 1.0 / j

            # Add contribution to accumulator
            acc += ni * t

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
        from ...utils.exceptions import TheoreticalInconsistencyError

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
        from ...utils.exceptions import TheoreticalInconsistencyError

        raise TheoreticalInconsistencyError(
            "Cross-entropy is not implemented for Bonachela estimator due to "
            "theoretical inconsistencies in applying bias corrections from "
            "different distributions."
        )
