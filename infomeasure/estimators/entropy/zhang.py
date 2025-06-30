"""Module for the Zhang entropy estimator."""

from numpy import log, array, arange, sum as np_sum, where

from infomeasure.estimators.base import DiscreteHEstimator


def _compute_vectorized_bias_correction_terms(N, max_k, valid_counts):
    """Compute vectorized bias correction terms for Zhang entropy estimation.

    Calculates the cumulative product factors and their harmonic-weighted sums
    that form the core bias correction components in the Zhang entropy formula.
    This implements the vectorized computation of the inner summation:
    sum(t1/k) where t1 is the cumulative product of bias correction factors.

    Parameters
    ----------
    valid_counts : ndarray
        Array of valid count values (counts > 0 and counts < N).
    N : int
        Total number of observations.

    Returns
    -------
    ndarray
        Array of bias correction terms corresponding to each valid count.
        Shape: (len(valid_counts),)
    """
    k_values = arange(1, max_k + 1)  # Shape: (max_k,)
    # Create mask for valid k values for each count | Shape: (len(valid_counts), max_k)
    valid_k_mask = k_values[None, :] <= (N - valid_counts[:, None])

    # Calculate factors for each (count, k) pair | Shape: (len(valid_counts), max_k)
    factors = 1.0 - (valid_counts[:, None] - 1.0) / (N - k_values[None, :])

    # Apply mask to factors - Set invalid factors to 1.0 (neutral for product)
    factors = where(valid_k_mask, factors, 1.0)

    # Calculate cumulative products along k dimension for each count
    t1_matrix = factors.cumprod(axis=1)  # Shape: (len(valid_counts), max_k)

    # Apply mask again and calculate t2 for each count
    t1_masked = where(valid_k_mask, t1_matrix, 0.0)  # Set invalid t1 values to 0.0
    reciprocal_k = 1.0 / k_values[None, :]  # Shape: (1, max_k)
    # Calculate t2 = sum(t1/k) for each count
    t2_values = np_sum(t1_masked * reciprocal_k, axis=1)  # Shape: (len(valid_counts),)

    return t2_values


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
        N = self.data[0].N

        # Filter out invalid counts (0 or >= N)
        valid_mask = (counts > 0) & (counts < N)
        valid_counts = counts[valid_mask]

        if len(valid_counts) == 0:
            return 0.0

        # Vectorized computation
        # We need to handle different ranges for each count
        # This is more complex than Bonachela because the range depends on the count

        max_k = N - valid_counts.min()  # Maximum possible k value
        if max_k <= 0:
            return 0.0

        # Create k values array
        t2_values = _compute_vectorized_bias_correction_terms(N, max_k, valid_counts)

        # Calculate contributions
        contributions = t2_values * (valid_counts / N)  # Shape: (len(valid_counts),)

        # Sum all contributions
        ent = np_sum(contributions)

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
        # Get counts, unique values, and total observations
        counts = self.data[0].counts
        uniq_vals = self.data[0].uniq
        N = self.data[0].N

        # Filter out invalid counts (0 or >= N)
        valid_mask = (counts > 0) & (counts < N)
        valid_counts = counts[valid_mask]
        valid_uniq_vals = uniq_vals[valid_mask]

        # Create a mapping from unique values to their Zhang entropy contributions
        zhang_contributions = {}

        # Set contributions for invalid counts to 0.0
        for i, (uniq_val, count) in enumerate(zip(uniq_vals, counts)):
            if count == 0 or count >= N:
                zhang_contributions[uniq_val] = 0.0

        if len(valid_counts) > 0:
            # Vectorized computation for valid counts
            max_k = N - valid_counts.min()  # Maximum possible k value

            if max_k > 0:
                # Create k values array
                t2_values = _compute_vectorized_bias_correction_terms(
                    N, max_k, valid_counts
                )

                # Store contributions for valid unique values
                for uniq_val, t2 in zip(valid_uniq_vals, t2_values):
                    zhang_contributions[uniq_val] = t2
            else:
                # If max_k <= 0, set all valid contributions to 0.0
                for uniq_val in valid_uniq_vals:
                    zhang_contributions[uniq_val] = 0.0

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
