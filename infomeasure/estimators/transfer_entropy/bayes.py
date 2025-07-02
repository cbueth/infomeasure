"""Module for the Bayes transfer entropy estimator."""

from abc import ABC

from numpy import issubdtype, integer

from infomeasure.estimators.base import (
    TransferEntropyEstimator,
    ConditionalTransferEntropyEstimator,
)

from ..entropy.bayes import BayesEntropyEstimator
from infomeasure import Config
from infomeasure.utils.types import LogBaseType


class BaseBayesTEEstimator(ABC):
    r"""Base class for the Bayes transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    cond : array-like, optional
        The conditional data used to estimate the conditional transfer entropy.
    alpha : float | str
        The concentration parameter α of the Dirichlet prior.
        Either a float or a string specifying the choice of concentration parameter.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Assumed time taken by info to transfer from source to destination.
        Not compatible with the ``cond`` parameter / conditional TE.
        Alternatively called `offset`.
    step_size : int, optional
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int, optional
        Number of past observations to consider for the source and destination data.
    cond_hist_len : int, optional
        Number of past observations to consider for the conditional data.
        Only used for conditional transfer entropy.

    Raises
    ------
    ValueError
        If alpha is not a valid concentration parameter.
    """

    def __init__(
        self,
        source,
        dest,
        *,  # Enforce keyword-only arguments
        cond=None,
        alpha: float | str,
        K: int = None,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        offset: int = None,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        """Initialize the BaseBayesTEEstimator.

        Parameters
        ----------
        source, dest : array-like
            The source (X) and destination (Y) data used to estimate the transfer entropy.
        cond : array-like, optional
            The conditional data used to estimate the conditional transfer entropy.
        alpha : float | str
            The concentration parameter α of the Dirichlet prior.
            Either a float or a string specifying the choice of concentration parameter.
        K : int, optional
            The support size. If not provided, uses the observed support size.
        prop_time : int, optional
            Number of positions to shift the data arrays relative to each other (multiple of
            ``step_size``).
            Delay/lag/shift between the variables, representing propagation time.
            Assumed time taken by info to transfer from source to destination
            Not compatible with the ``cond`` parameter / conditional TE.
            Alternatively called `offset`.
        step_size : int, optional
            Step size between elements for the state space reconstruction.
        src_hist_len, dest_hist_len : int, optional
            Number of past observations to consider for the source and destination data.
        cond_hist_len : int, optional
            Number of past observations to consider for the conditional data.
            Only used for conditional transfer entropy.

        """
        if cond is None:
            super().__init__(
                source,
                dest,
                prop_time=prop_time,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                step_size=step_size,
                offset=offset,
                base=base,
                **kwargs,
            )
        else:
            super().__init__(
                source,
                dest,
                cond=cond,
                step_size=step_size,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                cond_hist_len=cond_hist_len,
                prop_time=prop_time,
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


class BayesTEEstimator(BaseBayesTEEstimator, TransferEntropyEstimator):
    r"""Estimator for the Bayes transfer entropy.

    Bayesian transfer entropy estimator using Dirichlet prior with concentration
    parameter α. Provides principled handling of sparse data through Bayesian
    probability estimates.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    alpha : float | str
        The concentration parameter α of the Dirichlet prior.
        Either a float or a string specifying the choice of concentration parameter.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Assumed time taken by info to transfer from source to destination.
        Alternatively called `offset`.
    step_size : int, optional
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int, optional
        Number of past observations to consider for the source and destination data.

    Notes
    -----
    This estimator uses Bayesian probability estimates with a Dirichlet prior
    to compute transfer entropy through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_te_from_entropy)
    not a dedicated implementation as other TE might have.

    See Also
    --------
    infomeasure.estimators.entropy.bayes.BayesEntropyEstimator
        Bayesian entropy estimator with Dirichlet prior.
    """

    def _calculate(self):
        """Estimate the Bayes transfer entropy."""
        return self._generic_te_from_entropy(
            estimator=BayesEntropyEstimator,
            kwargs={"alpha": self.alpha, "K": self.K, "base": self.base},
        )


class BayesCTEEstimator(BaseBayesTEEstimator, ConditionalTransferEntropyEstimator):
    r"""Estimator for the Bayes conditional transfer entropy.

    Bayesian conditional transfer entropy estimator using Dirichlet prior with
    concentration parameter α. Provides principled handling of sparse data through
    Bayesian probability estimates.

    Attributes
    ----------
    source, dest, cond : array-like
        The source (X), destination (Y), and conditional (Z) data used to estimate the
        conditional transfer entropy.
    alpha : float | str
        The concentration parameter α of the Dirichlet prior.
        Either a float or a string specifying the choice of concentration parameter.
    K : int, optional
        The support size. If not provided, uses the observed support size.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len, cond_hist_len : int
        Number of past observations to consider for the source, destination,
        and conditional data.

    Notes
    -----
    This estimator uses Bayesian probability estimates with a Dirichlet prior
    to compute conditional transfer entropy through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_cte_from_entropy)
    not a dedicated implementation as other TE might have.

    See Also
    --------
    infomeasure.estimators.entropy.bayes.BayesEntropyEstimator
        Bayesian entropy estimator with Dirichlet prior.
    """

    def _calculate(self):
        """Estimate the Bayes conditional transfer entropy."""
        return self._generic_cte_from_entropy(
            estimator=BayesEntropyEstimator,
            kwargs={"alpha": self.alpha, "K": self.K, "base": self.base},
        )
