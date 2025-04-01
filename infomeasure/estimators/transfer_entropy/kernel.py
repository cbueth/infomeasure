"""Module for the kernel-based transfer entropy estimator."""

from abc import ABC

from numpy import zeros

from ..base import (
    PValueMixin,
    EffectiveValueMixin,
    TransferEntropyEstimator,
    ConditionalTransferEntropyEstimator,
)
from ..utils.kde import kde_probability_density_function
from ..utils.te_slicing import te_observations, cte_observations
from ... import Config
from ...utils.types import LogBaseType


class BaseKernelTEEstimator(ABC):
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
    """

    def __init__(
        self,
        source,
        dest,
        cond=None,
        *,  # all following parameters are keyword-only
        bandwidth: float | int = None,
        kernel: str = None,
        prop_time: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        cond_hist_len: int = 1,
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
        """
        if cond is None:
            super().__init__(
                source,
                dest,
                prop_time=prop_time,
                step_size=step_size,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                offset=offset,
                base=base,
            )
        else:
            super().__init__(
                source,
                dest,
                cond,
                step_size=step_size,
                src_hist_len=src_hist_len,
                dest_hist_len=dest_hist_len,
                cond_hist_len=cond_hist_len,
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
        local_te_values = zeros(len(joint_space_data))

        # Compute KDE for each term directly using slices
        for i in range(len(joint_space_data)):
            # TODO: Vectorized version speeds up kde computation,
            #  should outweigh the early stopping in this loop
            # g(x_i^{(l)}, y_i^{(k)}, y_{i+1})
            p_x_past_y_past_y_future = kde_probability_density_function(
                joint_space_data, self.bandwidth, joint_space_data[i], self.kernel
            )
            if p_x_past_y_past_y_future == 0:
                continue
            # g(y_i^{(k)})
            p_y_past = kde_probability_density_function(
                dest_past_embedded, self.bandwidth, dest_past_embedded[i], self.kernel
            )
            numerator = p_x_past_y_past_y_future * p_y_past
            if numerator <= 0:
                continue
            # g(x_i^{(l)}, y_i^{(k)})
            p_xy_past = kde_probability_density_function(
                marginal_1_space_data,
                self.bandwidth,
                marginal_1_space_data[i],
                self.kernel,
            )
            if p_xy_past == 0:
                continue
            # g(y_i^{(k)}, y_{i+1})
            p_y_past_y_future = kde_probability_density_function(
                marginal_2_space_data,
                self.bandwidth,
                marginal_2_space_data[i],
                self.kernel,
            )
            denominator = p_xy_past * p_y_past_y_future
            if denominator <= 0:
                continue

            local_te_values[i] = self._log_base(numerator / denominator)

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
        local_cte_values = zeros(len(joint_space_data))

        # Compute KDE for each term directly using slices
        for i in range(len(joint_space_data)):
            # g(x_i^{(l)}, z_i^{(m)}, y_i^{(k)}, y_{i+1})
            p_x_history_cond_y_history_y_future = kde_probability_density_function(
                joint_space_data, self.bandwidth, joint_space_data[i], self.kernel
            )
            if p_x_history_cond_y_history_y_future == 0:
                continue
            # g(y_i^{(k)}, z_i^{(m)})
            p_y_history_cond = kde_probability_density_function(
                dest_past_embedded, self.bandwidth, dest_past_embedded[i], self.kernel
            )
            numerator = p_x_history_cond_y_history_y_future * p_y_history_cond
            if numerator <= 0:
                continue
            # g(x_i^{(l)}, z_i^{(m)}, y_i^{(k)})
            p_x_history_cond_y_history = kde_probability_density_function(
                marginal_1_space_data,
                self.bandwidth,
                marginal_1_space_data[i],
                self.kernel,
            )
            if p_x_history_cond_y_history == 0:
                continue
            # g(z_i^{(m)}, y_i^{(k)}, y_{i+1})
            p_cond_y_history_y_future = kde_probability_density_function(
                marginal_2_space_data,
                self.bandwidth,
                marginal_2_space_data[i],
                self.kernel,
            )
            denominator = p_x_history_cond_y_history * p_cond_y_history_y_future
            if denominator <= 0:
                continue

            local_cte_values[i] = self._log_base(numerator / denominator)

        return local_cte_values
