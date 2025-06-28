"""Module for the Zhang entropy estimator."""

from numpy import log, array

from infomeasure.estimators.base import DiscreteHEstimator


class ZhangEntropyEstimator(DiscreteHEstimator):
    r"""Zhang entropy estimator for discrete data.

    The Zhang estimator computes the Shannon entropy using the recommended definition
    from :cite:p:`grabchakAuthorshipAttributionUsing2013`:

    .. math::

        \hat{H}_Z = \sum_{i=1}^K \hat{p}_i \sum_{v=1}^{N - n_i} \frac{1}{v} \prod_{j=0}^{v-1} \left( 1 + \frac{1 - n_i}{N - 1 - j} \right)

    where :math:`\hat{p}_i` are the empirical probabilities, :math:`n_i` are the counts
    for each unique value, :math:`K` is the number of unique values, and :math:`N` is
    the total number of observations.

    The actual algorithm implementation follows the fast calculation approach from
    :cite:p:`lozanoFastCalculationEntropy2017`.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    """

    def _simple_entropy(self):
        """Calculate the Zhang entropy of the data.

        Returns
        -------
        float
            The calculated Zhang entropy.
        """
        # Get counts and total observations
        counts = self.data[0].counts
        N = int(self.data[0].N)

        ent = 0.0

        # Iterate over each unique value and its count
        for count in counts:
            count = int(count)  # Ensure count is an integer
            # Skip if count is 0 or greater than N-1 (edge case)
            if count == 0 or count >= N:
                continue

            # Calculate the inner sum with product
            t1 = 1.0
            t2 = 0.0

            for k in range(1, N - count + 1):
                t1 *= 1.0 - (count - 1.0) / (N - k)
                t2 += t1 / k

            # Add contribution to entropy
            ent += t2 * (count / N)

        # Convert to the desired base if needed
        if self.base != "e":
            ent /= log(self.base)

        return ent

    def _extract_local_values(self):
        """Calculate local Zhang entropy values for each data point.

        Returns
        -------
        ndarray[float]
            The calculated local values of Zhang entropy.
        """
        from numpy import zeros

        # Get the distribution dictionary and original data
        counts = self.data[0].counts
        N = int(self.data[0].N)

        # Create a mapping from unique values to their Zhang entropy contributions
        zhang_contributions = {}

        # Calculate Zhang entropy contribution for each unique value
        for i, (uniq_val, count) in enumerate(zip(self.data[0].uniq, counts)):
            count = int(count)
            if count == 0 or count >= N:
                zhang_contributions[uniq_val] = 0.0
                continue

            # Calculate the inner sum with product for this count
            t1 = 1.0
            t2 = 0.0

            for k in range(1, N - count + 1):
                t1 *= 1.0 - (count - 1.0) / (N - k)
                t2 += t1 / k

            # Store the contribution per occurrence
            zhang_contributions[uniq_val] = t2

        # Map each data point to its local Zhang entropy value
        local_values = array([zhang_contributions[val] for val in self.data[0].data])

        # Convert to the desired base if needed
        if self.base != "e":
            local_values /= log(self.base)

        return local_values

    def _cross_entropy(self):
        """Calculate cross-entropy between two distributions using Zhang estimator.

        Returns
        -------
        float
            The calculated cross-entropy.
        """
        from ...utils.exceptions import TheoreticalInconsistencyError

        raise TheoreticalInconsistencyError(
            "Cross-entropy is not implemented for Zhang estimator due to "
            "theoretical inconsistencies in applying bias corrections from "
            "different distributions."
        )
