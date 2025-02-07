"""Module for the discrete mutual information estimator."""

from abc import ABC
from numpy import abs as np_abs, ndarray
from numpy import clip, finfo, int64, ravel, where
from scipy.sparse import find as sp_find
from scipy.stats.contingency import crosstab

from ..entropy import DiscreteEntropyEstimator
from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType
from ..base import (
    EffectiveValueMixin,
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)


class BaseDiscreteMIEstimator(ABC):
    """Base class for discrete mutual information estimators.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the (conditional) mutual information.
    data_z : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``data_z`` parameter / conditional MI.
    """

    def __init__(
        self,
        data_x,
        data_y,
        data_z=None,
        *,  # all following parameters are keyword-only
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the BaseDiscreteMIEstimator.

        Parameters
        ----------
        data_x, data_y : array-like
            The data used to estimate the (conditional) mutual information.
        data_z : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``data_z`` parameter / conditional MI.

        """
        self.data_y: ndarray = None
        self.data_x: ndarray = None
        if data_z is None:
            super().__init__(data_x, data_y, offset=offset, normalize=False, base=base)
        else:
            super().__init__(
                data_x, data_y, data_z, offset=offset, normalize=False, base=base
            )
        if self.data_x.dtype.kind == "f" or self.data_y.dtype.kind == "f":
            logger.warning(
                "The data looks like a float array ("
                f"data_x: {self.data_x.dtype}, data_y: {self.data_y.dtype}). "
                "Make sure it is properly symbolized or discretized "
                "for the mutual information estimation."
            )
        if hasattr(self, "data_z") and self.data_z.dtype.kind == "f":
            logger.warning(
                "The conditional data looks like a float array ("
                f"{self.data_z.dtype}). "
                "Make sure it is properly symbolized or discretized "
                "for the conditional mutual information estimation."
            )


class DiscreteMIEstimator(
    BaseDiscreteMIEstimator, EffectiveValueMixin, MutualInformationEstimator
):
    """Estimator for the discrete mutual information.

    Attributes
    ----------
    data_x, data_y : array-like
        The data used to estimate the mutual information.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    """

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


class DiscreteCMIEstimator(
    BaseDiscreteMIEstimator, ConditionalMutualInformationEstimator
):
    """Estimator for the discrete conditional mutual information.

    Attributes
    ----------
    data_x, data_y, data_z : array-like
        The data used to estimate the conditional mutual information.
    """

    def _calculate(self):
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        float
            The calculated conditional mutual information.
        """
        return self._generic_cmi_from_entropy(
            estimator=DiscreteEntropyEstimator, kwargs=dict(base=self.base)
        )
