"""Module for the discrete mutual information estimator."""

from numpy import abs as np_abs
from numpy import clip, finfo, int64, ravel, where
from scipy.sparse import find as sp_find
from scipy.stats.contingency import crosstab

from ... import Config
from ...utils.types import LogBaseType
from ..base import EffectiveValueMixin, MutualInformationEstimator


class DiscreteMIEstimator(EffectiveValueMixin, MutualInformationEstimator):
    """Estimator for discrete mutual information.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    base : int | float | "e", optional
        The logarithm base for the mutual information calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def __init__(
        self, data_x, data_y, offset: int = 0, base: LogBaseType = Config.get("base")
    ):
        """Initialize the estimator with specific offset.

        Parameters
        ----------
        offset : int, optional
            Number of positions to shift the data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
        """
        super().__init__(data_x, data_y, offset=offset, normalize=False, base=base)

    def _calculate(self):
        """Calculate the mutual information of the data.

        The approach relies on the contingency table of the two variables.
        Instead of calculating the full outer product, only the non-zero elements are
        considered.
        Code adapted from
        the :func:`mutual_info_score() <sklearn.metrics.mutual_info_score>` function in
        scikit-learn.

        Returns
        -------
        float
            The calculated mutual information.
        """
        # Contingency Table - sparse matrix
        contingency_coo = crosstab(self.data_x, self.data_y, sparse=True).count
        # Non-zero indices and values
        nzx, nzy, nzv = sp_find(contingency_coo)

        # Normalized contingency table (joint probability)
        contingency_sum = contingency_coo.sum()
        contingency_nm = nzv / contingency_sum
        # Marginal probabilities
        pi = ravel(contingency_coo.sum(axis=1))
        pj = ravel(contingency_coo.sum(axis=0))

        # Early return if any of the marginal entropies is zero
        if pi.size == 1 or pj.size == 1:
            return 0.0

        # Logarithm of the non-zero elements
        log_contingency_nm = self._log_base(nzv)

        # Calculate the expected logarithm values for the outer product of marginal
        # probabilities, only for non-zero entries.
        outer = pi.take(nzx).astype(int64, copy=False) * pj.take(nzy).astype(
            int64, copy=False
        )
        log_outer = (
            -self._log_base(outer) + self._log_base(pi.sum()) + self._log_base(pj.sum())
        )
        # Combine the terms to calculate the mutual information
        mi = (
            contingency_nm * (log_contingency_nm - self._log_base(contingency_sum))
            + contingency_nm * log_outer
        )
        # Filter values below floating point precision
        mi = where(np_abs(mi) < finfo(mi.dtype).eps, 0.0, mi)
        # Clip negative values to zero
        return clip(mi.sum(), 0.0, None)
