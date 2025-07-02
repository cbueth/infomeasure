"""Module for the Ansb mutual information estimator."""

from abc import ABC

from numpy import issubdtype, integer

from infomeasure.estimators.base import (
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)

from ..entropy.ansb import AnsbEntropyEstimator
from infomeasure import Config
from infomeasure.utils.types import LogBaseType


class BaseAnsbMIEstimator(ABC):
    r"""Base class for Ansb mutual information estimators.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    undersampled : float, default=0.1
        Maximum allowed ratio N/K to consider data sufficiently undersampled.
        A warning is issued if this threshold is exceeded.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Not compatible with the ``cond`` parameter / conditional MI.
    """

    def __init__(
        self,
        *data,
        cond=None,
        undersampled: float = 0.1,
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        r"""Initialize the Ansb estimator with specific parameters.

        Parameters
        ----------
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        undersampled : float, default=0.1
            Maximum allowed ratio N/K to consider data sufficiently undersampled.
            A warning is issued if this threshold is exceeded.
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
        if undersampled < 0:
            raise ValueError("The `undersampled` parameter must not be negative.")

        self.undersampled = undersampled


class AnsbMIEstimator(BaseAnsbMIEstimator, MutualInformationEstimator):
    r"""Estimator for the Ansb mutual information.

    Ansb mutual information estimator using the entropy combination formula.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    undersampled : float, default=0.1
        Maximum allowed ratio N/K to consider data sufficiently undersampled.
        A warning is issued if this threshold is exceeded.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.

    Notes
    -----
    This estimator uses the Ansb entropy estimator to compute mutual
    information through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_mi_from_entropy)
    not a dedicated implementation as other MI might have.

    See Also
    --------
    infomeasure.estimators.entropy.ansb.AnsbEntropyEstimator
        Ansb entropy estimator.
    """

    def _calculate(self):
        """Calculate the mutual information of the data.

        Returns
        -------
        float
            Ansb mutual information of the data.
        """

        return self._generic_mi_from_entropy(
            estimator=AnsbEntropyEstimator,
            kwargs={"undersampled": self.undersampled, "base": self.base},
        )


class AnsbCMIEstimator(BaseAnsbMIEstimator, ConditionalMutualInformationEstimator):
    r"""Estimator for the conditional Ansb mutual information.

    Ansb conditional mutual information estimator using the entropy combination formula.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    undersampled : float, default=0.1
        Maximum allowed ratio N/K to consider data sufficiently undersampled.
        A warning is issued if this threshold is exceeded.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.

    Notes
    -----
    This estimator uses the Ansb entropy estimator to compute conditional
    mutual information through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_cmi_from_entropy)
    not a dedicated implementation as other MI might have.

    See Also
    --------
    infomeasure.estimators.entropy.ansb.AnsbEntropyEstimator
        Ansb entropy estimator.
    """

    def _calculate(self):
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        float
            Conditional Ansb mutual information of the data.
        """
        return self._generic_cmi_from_entropy(
            estimator=AnsbEntropyEstimator,
            kwargs={"undersampled": self.undersampled, "base": self.base},
        )
