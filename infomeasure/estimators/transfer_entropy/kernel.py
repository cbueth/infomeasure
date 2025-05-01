"""Module for the kernel-based transfer entropy estimator."""

from abc import ABC

from numpy import isinf, isnan

from ... import Config
from ...utils.config import logger
from ...utils.types import LogBaseType
from ..base import (
    ConditionalTransferEntropyEstimator,
    EffectiveValueMixin,
    PValueMixin,
    TransferEntropyEstimator,
    WorkersMixin,
)
from ..utils.kde import kde_probability_density_function
from ..utils.te_slicing import cte_observations, te_observations


class BaseKernelTEEstimator(WorkersMixin, ABC):
    """Base class for transfer entropy using Kernel Density Estimation (KDE).

    Attributes
    ----------
    source, dest : array-like
        The source (X) and destination (Y) data used to estimate the transfer entropy.
    cond : array-like, optional
        The conditional data used to estimate the conditional transfer entropy.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
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
    workers : int, optional
       Number of workers to use for parallel processing.
       Default is 1, meaning no parallel processing.
       If set to -1, all available CPU cores will be used.
    """

    def __init__(
        self,
        source,
        dest,
        *,  # Enforce keyword-only arguments
        cond=None,
        bandwidth: float | int = None,
        kernel: str = None,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
        workers: int = 1,
        offset: int = None,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the BaseKernelTEEstimator.

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
        workers : int, optional
           Number of workers to use for parallel processing.
        """
        if cond is None:
            super().__init__(
                source,
                dest,
                prop_time=prop_time,
                step_size=step_size,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                workers=workers,
                offset=offset,
                base=base,
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
                workers=workers,
                prop_time=prop_time,
                offset=offset,
                base=base,
            )
        self.bandwidth = bandwidth
        self.kernel = kernel


class KernelTEEstimator(
    BaseKernelTEEstimator, PValueMixin, EffectiveValueMixin, TransferEntropyEstimator
):
    """Estimator for transfer entropy using Kernel Density Estimation (KDE).

    Attributes
    ----------
    source, dest : array-like
        The source (X) and destination (Y) data used to estimate the transfer entropy.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
    prop_time : int, optional
        Number of positions to shift the data arrays relative to each other (multiple of
        ``step_size``).
        Delay/lag/shift between the variables, representing propagation time.
        Assumed time taken by info to transfer from source to destination.
        Alternatively called `offset`.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.

    Notes
    -----
    A small ``bandwidth`` can lead to under-sampling,
    while a large ``bandwidth`` may over-smooth the data, obscuring details.
    """

    def _calculate(self):
        """Calculate the transfer entropy of the data.

        Returns
        -------
        local_te_values : array
            Local transfer entropy values.
        """
        # Prepare multivariate data arrays for KDE: Numerators
        (
            joint_space_data,
            dest_past_embedded,
            marginal_1_space_data,
            marginal_2_space_data,
        ) = te_observations(
            self.source,
            self.dest,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            step_size=self.step_size,
            permute_src=self.permute_src,
            resample_src=self.resample_src,
        )

        # Compute densities in vectorized manner
        # g(x_i^{(l)}, y_i^{(k)}, y_{i+1})
        logger.debug(
            "Calculating densities for...\n1/4 g(x_i^{(l)}, y_i^{(k)}, y_{i+1})"
        )
        p_x_past_y_past_y_future = kde_probability_density_function(
            joint_space_data, self.bandwidth, kernel=self.kernel, workers=self.n_workers
        )
        # g(y_i^{(k)})
        logger.debug("2/4 g(y_i^{(k)})")
        p_y_past = kde_probability_density_function(
            dest_past_embedded,
            self.bandwidth,
            kernel=self.kernel,
            workers=self.n_workers,
        )
        # g(x_i^{(l)}, y_i^{(k)})
        logger.debug("3/4 g(x_i^{(l)}, y_i^{(k)})")
        p_xy_past = kde_probability_density_function(
            marginal_1_space_data,
            self.bandwidth,
            kernel=self.kernel,
            workers=self.n_workers,
        )
        # g(y_i^{(k)}, y_{i+1})
        logger.debug("4/4 g(y_i^{(k)}, y_{i+1})")
        p_y_past_y_future = kde_probability_density_function(
            marginal_2_space_data,
            self.bandwidth,
            kernel=self.kernel,
            workers=self.n_workers,
        )

        local_te_values = self._log_base(
            (p_x_past_y_past_y_future * p_y_past) / (p_y_past_y_future * p_xy_past)
        )
        # where inf/nan set to zero
        local_te_values[isinf(local_te_values) | isnan(local_te_values)] = 0.0

        return local_te_values


class KernelCTEEstimator(BaseKernelTEEstimator, ConditionalTransferEntropyEstimator):
    """Estimator for conditional transfer entropy using Kernel Density Estimation (KDE).

    Attributes
    ----------
    source, dest, cond : array-like
        The source (X), destination (Y), and conditional (Z) data used to estimate the
        conditional transfer entropy.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.estimators.utils.kde.kde_probability_density_function>`.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len, cond_hist_len : int, optional
        Number of past observations to consider for the source, destination,
        and conditional data.
    prop_time : int, optional
        Not compatible with the ``cond`` parameter / conditional TE.

    Notes
    -----
    A small ``bandwidth`` can lead to under-sampling,
    while a large ``bandwidth`` may over-smooth the data, obscuring details.
    """

    def _calculate(self):
        """Calculate the conditional transfer entropy of the data.

        Returns
        -------
        local_cte_values : array
            Local conditional transfer entropy values.
        """
        # Prepare multivariate data arrays for KDE: Numerators
        (
            joint_space_data,
            dest_past_embedded,
            marginal_1_space_data,
            marginal_2_space_data,
        ) = cte_observations(
            self.source,
            self.dest,
            self.cond,
            src_hist_len=self.src_hist_len,
            dest_hist_len=self.dest_hist_len,
            cond_hist_len=self.cond_hist_len,
            step_size=self.step_size,
        )

        # Calculate densities in vectorized manner
        # g(x_i^{(l)}, z_i^{(m)}, y_i^{(k)}, y_{i+1})
        logger.debug(
            "Calculating densities for...\n"
            "1/4 g(x_i^{(l)}, z_i^{(m)}, y_i^{(k)}, y_{i+1})"
        )
        p_x_history_cond_y_history_y_future = kde_probability_density_function(
            joint_space_data, self.bandwidth, kernel=self.kernel, workers=self.n_workers
        )
        # g(y_i^{(k)}, z_i^{(m)})
        logger.debug("2/4 g(y_i^{(k)}, z_i^{(m)})")
        p_y_history_cond = kde_probability_density_function(
            dest_past_embedded,
            self.bandwidth,
            kernel=self.kernel,
            workers=self.n_workers,
        )
        # g(x_i^{(l)}, z_i^{(m)}, y_i^{(k)})
        logger.debug("3/4 g(x_i^{(l)}, z_i^{(m)}, y_i^{(k)})")
        p_x_history_cond_y_history = kde_probability_density_function(
            marginal_1_space_data,
            self.bandwidth,
            kernel=self.kernel,
            workers=self.n_workers,
        )
        # g(z_i^{(m)}, y_i^{(k)}, y_{i+1})
        logger.debug("4/4 g(z_i^{(m)}, y_i^{(k)}, y_{i+1})")
        p_cond_y_history_y_future = kde_probability_density_function(
            marginal_2_space_data,
            self.bandwidth,
            kernel=self.kernel,
            workers=self.n_workers,
        )

        local_cte_values = self._log_base(
            (p_x_history_cond_y_history_y_future * p_y_history_cond)
            / (p_x_history_cond_y_history * p_cond_y_history_y_future)
        )
        # where inf/nan set to zero
        local_cte_values[isinf(local_cte_values) | isnan(local_cte_values)] = 0.0

        return local_cte_values
