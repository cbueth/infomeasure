"""Module for the Bayes mutual information estimator."""

from abc import ABC

from numpy import issubdtype, integer

from infomeasure.estimators.base import (
    MutualInformationEstimator,
    ConditionalMutualInformationEstimator,
)

from ..entropy.bayes import BayesEntropyEstimator
from infomeasure import Config
from infomeasure.utils.types import LogBaseType


class BaseBayesMIEstimator(ABC):
    r"""Base class for Bayes mutual information estimators.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the (conditional) mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
        For conditional mutual information, only two data arrays are allowed.
    cond : array-like, optional
        The conditional data used to estimate the conditional mutual information.
    alpha : float | str
        The concentration parameter α of the Dirichlet prior.
        Either a float or a string specifying the choice of concentration parameter.
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
        alpha: float | str,
        K: int = None,
        offset: int = 0,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        r"""Initialize the Bayes estimator with specific parameters.

        Parameters
        ----------
        *data : array-like, shape (n_samples,)
            The data used to estimate the (conditional) mutual information.
            You can pass an arbitrary number of data arrays as positional arguments.
            For conditional mutual information, only two data arrays are allowed.
        cond : array-like, optional
            The conditional data used to estimate the conditional mutual information.
        alpha : float | str
            The concentration parameter α of the Dirichlet prior.
            Either a float or a string specifying the choice of concentration parameter.
        K : int, optional
            The support size. If not provided, uses the observed support size.
        offset : int, optional
            Number of positions to shift the X and Y data arrays relative to each other.
            Delay/lag/shift between the variables. Default is no shift.
            Not compatible with the ``cond`` parameter / conditional MI.

        Raises
        ------
        ValueError
            If alpha is not a valid concentration parameter.
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
        # Parameter validation
        if not isinstance(alpha, (int, float, str)):
            raise ValueError("The alpha parameter must be a number or string.")
        if K is not None and (not isinstance(K, int) or K <= 0):
            raise ValueError("The K parameter must be a positive integer.")

        self.alpha = alpha
        self.K = K


class BayesMIEstimator(BaseBayesMIEstimator, MutualInformationEstimator):
    r"""Estimator for the Bayes mutual information.

    Bayesian mutual information estimator using Dirichlet prior with concentration
    parameter α. Provides principled handling of sparse data through Bayesian
    probability estimates.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    alpha : float | str
        The concentration parameter α of the Dirichlet prior.
        Either a float or a string specifying the choice of concentration parameter.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.

    Notes
    -----
    This estimator uses Bayesian probability estimates with a Dirichlet prior
    to compute mutual information through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_mi_from_entropy)
    not a dedicated implementation as other MI might have.

    See Also
    --------
    infomeasure.estimators.entropy.bayes.BayesEntropyEstimator
        Bayesian entropy estimator with Dirichlet prior.
    """

    def _calculate(self):
        """Calculate the mutual information of the data.

        Returns
        -------
        float
            Bayes mutual information of the data.
        """

        return self._generic_mi_from_entropy(
            estimator=BayesEntropyEstimator,
            kwargs={"alpha": self.alpha, "K": self.K, "base": self.base},
        )


class BayesCMIEstimator(BaseBayesMIEstimator, ConditionalMutualInformationEstimator):
    r"""Estimator for the conditional Bayes mutual information.

    Bayesian conditional mutual information estimator using Dirichlet prior with
    concentration parameter α. Provides principled handling of sparse data through
    Bayesian probability estimates.

    Attributes
    ----------
    *data : array-like, shape (n_samples,)
        The data used to estimate the conditional mutual information.
        You can pass an arbitrary number of data arrays as positional arguments.
    cond : array-like
        The conditional data used to estimate the conditional mutual information.
    alpha : float | str
        The concentration parameter α of the Dirichlet prior.
        Either a float or a string specifying the choice of concentration parameter.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.

    Notes
    -----
    This estimator uses Bayesian probability estimates with a Dirichlet prior
    to compute conditional mutual information through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_cmi_from_entropy)
    not a dedicated implementation as other MI might have.

    See Also
    --------
    infomeasure.estimators.entropy.bayes.BayesEntropyEstimator
        Bayesian entropy estimator with Dirichlet prior.
    """

    def _calculate(self):
        """Calculate the conditional mutual information of the data.

        Returns
        -------
        float
            Conditional Bayes mutual information of the data.
        """
        return self._generic_cmi_from_entropy(
            estimator=BayesEntropyEstimator,
            kwargs={"alpha": self.alpha, "K": self.K, "base": self.base},
        )
