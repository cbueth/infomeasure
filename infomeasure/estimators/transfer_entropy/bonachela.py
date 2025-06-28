"""Module for the Bonachela transfer entropy estimator."""

from abc import ABC

from numpy import issubdtype, integer

from infomeasure.estimators.base import (
    TransferEntropyEstimator,
    ConditionalTransferEntropyEstimator,
)

from ..entropy.bonachela import BonachelaEntropyEstimator
from infomeasure import Config
from infomeasure.utils.types import LogBaseType


class BaseBonachelaTEEstimator(ABC):
    r"""Base class for the Bonachela transfer entropy.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.
    cond : array-like, optional
        The conditional data used to estimate the conditional transfer entropy.

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
    """

    def __init__(
        self,
        source,
        dest,
        *,  # Enforce keyword-only arguments
        cond=None,
        
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        offset: int = None,
        base: LogBaseType = Config.get("base"),
        **kwargs,
    ):
        """Initialize the BaseBonachelaTEEstimator.

        Parameters
        ----------
        source, dest : array-like
            The source (X) and destination (Y) data used to estimate the transfer entropy.
        cond : array-like, optional
            The conditional data used to estimate the conditional transfer entropy.

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



class BonachelaTEEstimator(BaseBonachelaTEEstimator, TransferEntropyEstimator):
    r"""Estimator for the Bonachela transfer entropy.

    Bonachela transfer entropy estimator using the entropy combination formula.

    Attributes
    ----------
    source, dest : array-like
        The source (X) and dest (Y) data used to estimate the transfer entropy.

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
    This estimator uses the Bonachela entropy estimator to compute transfer 
    entropy through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_te_from_entropy)
    not a dedicated implementation as other TE might have.

    See Also
    --------
    infomeasure.estimators.entropy.bonachela.BonachelaEntropyEstimator
        Bonachela entropy estimator.
    """

    def _calculate(self):
        """Estimate the Bonachela transfer entropy."""
        return self._generic_te_from_entropy(
            estimator=BonachelaEntropyEstimator,
            kwargs={"base": self.base},
        )


class BonachelaCTEEstimator(BaseBonachelaTEEstimator, ConditionalTransferEntropyEstimator):
    r"""Estimator for the Bonachela conditional transfer entropy.

    Bonachela conditional transfer entropy estimator using the entropy combination formula.

    Attributes
    ----------
    source, dest, cond : array-like
        The source (X), destination (Y), and conditional (Z) data used to estimate the
        conditional transfer entropy.

    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len, cond_hist_len : int
        Number of past observations to consider for the source, destination,
        and conditional data.

    Notes
    -----
    This estimator uses the Bonachela entropy estimator to compute conditional 
    transfer entropy through the entropy combination formula.

    Note that the entropy combination formula is used (_generic_cte_from_entropy)
    not a dedicated implementation as other TE might have.

    See Also
    --------
    infomeasure.estimators.entropy.bonachela.BonachelaEntropyEstimator
        Bonachela entropy estimator.
    """

    def _calculate(self):
        """Estimate the Bonachela conditional transfer entropy."""
        return self._generic_cte_from_entropy(
            estimator=BonachelaEntropyEstimator,
            kwargs={"base": self.base},
        )
