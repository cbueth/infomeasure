"""Module for the Zhang mutual information estimator."""

from abc import ABC

from numpy import issubdtype, integer

from infomeasure.estimators.base import (
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)

from ..entropy.zhang import ZhangEntropyEstimator
from infomeasure import Config
from infomeasure.utils.types import LogBaseType


class BaseZhangMIEstimator(ABC):
    r"""Base class for Zhang mutual information estimators.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.

    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``cond`` parameter / conditional MI.
    """

    def __init__(
        self,
        *data,
        cond=None,
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        r"""Initialize the Zhang estimator with specific parameters.

        Parameters
        ----------
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
            The conditional data used to estimate the conditional mutual information.

        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``cond`` parameter / conditional MI.
        """
        if cond is None:
            super().__init__(*data, offset=offset, base=base, **kwargs)
        else:
            super().__init__(
                *data,
                cond=cond,
                offset=offset,
                base=base,
                **kwargs,
            )


class ZhangMIEstimator(BaseZhangMIEstimator, MutualInformationEstimator):
    r"""Estimator for the Zhang mutual information.

    Zhang mutual information estimator using the entropy combination formula.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.

    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.

    Notes
    -----
    This estimator uses the Zhang entropy estimator to compute mutual
    information through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_mi_from_entropy)
    not a dedicated implementation as other MI might have.

    See Also
    --------
    infomeasure.estimators.entropy.zhang.ZhangEntropyEstimator
        Zhang entropy estimator.
    """

    def _calculate(self):
        """Calculate the mutual information of the data.

        Returns
        -------
        float
            Zhang mutual information of the data.
        """

        return self._generic_mi_from_entropy(
            estimator=ZhangEntropyEstimator,
            kwargs={"base": self.base},
        )


class ZhangCMIEstimator(BaseZhangMIEstimator, ConditionalMutualInformationEstimator):
    r"""Estimator for the conditional Zhang mutual information.

    Zhang conditional mutual information estimator using the entropy combination formula.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.

    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.

    Notes
    -----
    This estimator uses the Zhang entropy estimator to compute conditional
    mutual information through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_cmi_from_entropy)
    not a dedicated implementation as other MI might have.

    See Also
    --------
    infomeasure.estimators.entropy.zhang.ZhangEntropyEstimator
        Zhang entropy estimator.
    """

    def _calculate(self):
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        float
            Conditional Zhang mutual information of the data.
        """
        return self._generic_cmi_from_entropy(
            estimator=ZhangEntropyEstimator,
            kwargs={"base": self.base},
        )
