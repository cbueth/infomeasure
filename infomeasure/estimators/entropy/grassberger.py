"""Module for the discrete Grassberger entropy estimator."""

from numpy import asarray, log
from numpy import sum as np_sum
from scipy.special import digamma

from infomeasure.estimators.base import (
    DistributionMixin,
    EntropyEstimator,
    DiscreteMixin,
)
from ..utils.unique import unique_vals
from ... import Config
from ...utils.types import LogBaseType
from ...utils.exceptions import TheoreticalInconsistencyError


class GrassbergerEntropyEstimator(DistributionMixin, DiscreteMixin, EntropyEstimator):
    r"""Discrete Grassberger entropy estimator.

    .. math::

        \hat{H}_{\text{Gr88}} = \sum_i \frac{h_i}{H} \left(\log(N) - \psi(h_i) - \frac{(-1)^{h_i}}{n_i + 1}  \right)

    :math:`\hat{H}_{\text{Gr88}}` is the Grassberger entropy,
    where :math:`h_i` are the counts of unique values,
    :math:`H` is the total number of observations :math:`N`,
    :math:`\psi` is the digamma function,
    and :math:`n_i` are the counts (same as :math:`h_i`) :cite:p:`grassbergerFiniteSampleCorrections1988,grassbergerEntropyEstimatesInsufficient2008`.

    Attributes
    ----------
    *data : array-like
        The data used to estimate the entropy.
    """

    def __init__(self, *data, base: LogBaseType = Config.get("base")):
        """Initialize the GrassbergerEntropyEstimator."""
        super().__init__(*data, base=base)
        # warn if the data looks like a float array
        self._check_data_entropy()
        # reduce any joint space if applicable
        self._reduce_space()

    def _simple_entropy(self):
        """Calculate the Grassberger entropy of the data.

        Returns
        -------
        float
            The calculated entropy.
        """
        uniq, counts, self.dist_dict = unique_vals(self.data[0])
        N = len(self.data[0])

        # Create a mapping from unique values to their counts
        count_dict = dict(zip(uniq, counts))

        # Vectorized calculation of local values
        h_i = asarray([count_dict[val] for val in self.data[0]])
        local_values = log(N) - digamma(h_i) - ((-1) ** h_i) / (h_i + 1)

        # Convert to the requested base if needed
        if self.base != "e":
            local_values /= log(self.base)

        return local_values

    def _joint_entropy(self):
        """Calculate the joint Grassberger entropy of the data.

        Returns
        -------
        float
            The calculated joint entropy.
        """
        # The data has already been reduced to unique values of co-occurrences
        return self._simple_entropy()

    def _cross_entropy(self) -> float:
        """Calculate cross-entropy between two distributions.

        Raises
        ------
        TheoreticalInconsistencyError
            Cross-entropy is not theoretically sound for Grassberger estimator
            due to conceptual mismatch between bias correction and cross-entropy.
        """
        raise TheoreticalInconsistencyError(
            "Cross-entropy is not implemented for Grassberger estimator. "
            "The Grassberger correction is designed for bias correction in entropy "
            "estimation using count-based corrections, but cross-entropy mixes "
            "probabilities from one distribution with corrections from another, "
            "creating a theoretical inconsistency."
        )
