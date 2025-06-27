"""Module for the discrete mutual information estimator."""

from abc import ABC

from numpy import ndarray

from ..base import (
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)
from ..mixins import DiscreteMIMixin
from ..utils.discrete_interaction_information import (
    mutual_information_global,
    mutual_information_local,
    conditional_mutual_information_global,
    conditional_mutual_information_local,
)
from ... import Config
from ...utils.types import LogBaseType


class BaseDiscreteMIEstimator(DiscreteMIMixin, ABC):
    """Base class for discrete mutual information estimators.

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
        """Initialize the BaseDiscreteMIEstimator.

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
        self.data: tuple[ndarray] = None
        if cond is None:
            super().__init__(*data, offset=offset, normalize=False, base=base, **kwargs)
        else:
            super().__init__(
                *data, cond=cond, offset=offset, normalize=False, base=base, **kwargs
            )
        self._check_data_mi()


class DiscreteMIEstimator(BaseDiscreteMIEstimator, MutualInformationEstimator):
    """Estimator for the discrete mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
    """

    def _calculate(self):
        """Calculate the mutual information of the data.

        Returns
        -------
        float
            The calculated mutual information.
        """
        return mutual_information_global(*self.data, log_func=self._log_base)

    def _extract_local_values(self) -> ndarray[float]:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of mi.
        """
        return mutual_information_local(*self.data, log_func=self._log_base)


class DiscreteCMIEstimator(
    BaseDiscreteMIEstimator, ConditionalMutualInformationEstimator
):
    """Estimator for the discrete conditional mutual information.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    """

    def _calculate(self):
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        float
            The calculated conditional mutual information.
        """
        return conditional_mutual_information_global(
            *self.data, cond=self.cond, log_func=self._log_base
        )

    def _extract_local_values(self) -> ndarray:
        """Separately calculate the local values.

        Returns
        -------
        ndarray[float]
            The calculated local values of cmi.
        """
        return conditional_mutual_information_local(
            *self.data, cond=self.cond, log_func=self._log_base
        )
