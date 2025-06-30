"""Module for the Nsb mutual information estimator."""

from abc import ABC

from numpy import issubdtype, integer

from infomeasure.estimators.base import (
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)

from ..entropy.nsb import NsbEntropyEstimator
from infomeasure import Config
from infomeasure.utils.types import LogBaseType


class BaseNsbMIEstimator(ABC):
    r"""Base class for Nsb mutual information estimators.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``cond`` parameter / conditional MI.
    """

    def __init__(
        self,
        *data,
        cond=None,
        K: int = None,
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        r"""Initialize the Nsb estimator with specific parameters.

        Parameters
        ----------
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
            The conditional data used to estimate the conditional mutual information.
    K : int, optional
        The support size. If not provided, uses the observed support size.
        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``cond`` parameter / conditional MI.
        """
        if cond is None:
            super().__init__(
                *data, offset=offset, base=base, **kwargs
            )
        else:
            super().__init__(
                *data,
                cond=cond,
                offset=offset,
                base=base,
                **kwargs,
            )
        if K is not None and (not isinstance(K, int) or K <= 0):
            raise ValueError("The K parameter must be a positive integer.")
        
        self.K = K


class NsbMIEstimator(BaseNsbMIEstimator, MutualInformationEstimator):
    r"""Estimator for the Nsb mutual information.

    Nsb mutual information estimator using the entropy combination formula.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.

    Notes
    -----
    This estimator uses the Nsb entropy estimator to compute mutual 
    information through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_mi_from_entropy)
    not a dedicated implementation as other MI might have.

    See Also
    --------
    infomeasure.estimators.entropy.nsb.NsbEntropyEstimator
        Nsb entropy estimator.
    """

    def _calculate(self):
        """Calculate the mutual information of the data.

        Returns
        -------
        float
            Nsb mutual information of the data.
        """

        return self._generic_mi_from_entropy(
            estimator=NsbEntropyEstimator,
            kwargs={"K": self.K, "base": self.base},
        )


class NsbCMIEstimator(BaseNsbMIEstimator, ConditionalMutualInformationEstimator):
    r"""Estimator for the conditional Nsb mutual information.

    Nsb conditional mutual information estimator using the entropy combination formula.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.

    Notes
    -----
    This estimator uses the Nsb entropy estimator to compute conditional 
    mutual information through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_cmi_from_entropy)
    not a dedicated implementation as other MI might have.

    See Also
    --------
    infomeasure.estimators.entropy.nsb.NsbEntropyEstimator
        Nsb entropy estimator.
    """

    def _calculate(self):
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        float
            Conditional Nsb mutual information of the data.
        """
        return self._generic_cmi_from_entropy(
            estimator=NsbEntropyEstimator,
            kwargs={"K": self.K, "base": self.base},
        )
