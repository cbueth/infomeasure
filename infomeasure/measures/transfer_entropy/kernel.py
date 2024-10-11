"""Module for the kernel-based transfer entropy estimator."""

from numpy import nanmean, zeros

from ..utils.te_slicing import te_observations
from ... import Config
from ...utils.types import LogBaseType
from ..base import EffectiveTEMixin, TransferEntropyEstimator
from ..utils.kde import kde_probability_density_function


class KernelTEEstimator(EffectiveTEMixin, TransferEntropyEstimator):
    """Estimator for transfer entropy using Kernel Density Estimation (KDE).

    Attributes
    ----------
    source, dest : array-like
        The source (X) and destination (Y) data used to estimate the transfer entropy.
    bandwidth : float | int
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use, compatible with the KDE
        implementation :func:`kde_probability_density_function() <infomeasure.measures.utils.kde.kde_probability_density_function>`.
    offset : int, optional
        Number of positions to shift the data arrays relative to each other.
        Delay/lag/shift between the variables. Default is no shift.
        Assumed time taken by info to transfer from source to destination.
    step_size : int
        Step size between elements for the state space reconstruction.
    src_hist_len, dest_hist_len : int
        Number of past observations to consider for the source and destination data.
    base : int | float | "e", optional
        The logarithm base for the transfer entropy calculation.
        The default can be set
        with :func:`set_logarithmic_unit() <infomeasure.utils.config.Config.set_logarithmic_unit>`.
    """

    def __init__(
        self,
        source,
        dest,
        bandwidth: float | int,
        kernel: str,
        offset: int = 0,
        step_size: int = 1,
        src_hist_len: int = 1,
        dest_hist_len: int = 1,
        base: LogBaseType = Config.get("base"),
    ):
        """Initialize the estimator with the data and parameters.

        Parameters
        ----------
        bandwidth : float | int
            The bandwidth for the kernel.
        kernel : str
            Type of kernel to use, compatible with the KDE
            implementation :func:`kde_probability_density_function() <infomeasure.measures.utils.kde.kde_probability_density_function>`.
        src_hist_len, dest_hist_len : int
            Number of past observations to consider for the source and destination data.
        """
        super().__init__(
            source,
            dest,
            offset=offset,
            step_size=step_size,
            src_hist_len=src_hist_len,
            dest_hist_len=dest_hist_len,
            base=base,
        )
        self.bandwidth = bandwidth
        self.kernel = kernel

    def _calculate(self):
        """Calculate the transfer entropy of the data.

        Returns
        -------
        local_te_values : array
            Local transfer entropy values.
        average_te : float
            The average transfer entropy value.
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
        )
        local_te_values = zeros(len(joint_space_data))

        # Compute KDE for each term directly using slices
        for i in range(len(joint_space_data)):
            # g(y_{i+1}, y_i^{(k)}, x_i^{(l)})
            p_y_future_y_past_x_past = kde_probability_density_function(
                joint_space_data, joint_space_data[i], self.bandwidth, self.kernel
            )
            if p_y_future_y_past_x_past == 0:
                continue
            # g(y_i^{(k)})
            p_y_past = kde_probability_density_function(
                dest_past_embedded, dest_past_embedded[i], self.bandwidth, self.kernel
            )
            numerator = p_y_future_y_past_x_past * p_y_past
            if numerator <= 0:
                continue
            # g(y_i^{(k)}, x_i^{(l)})
            p_yx_past = kde_probability_density_function(
                marginal_1_space_data,
                marginal_1_space_data[i],
                self.bandwidth,
                self.kernel,
            )
            if p_yx_past == 0:
                continue
            # g(y_{i+1}, y_i^{(k)})
            p_y_future_y_past = kde_probability_density_function(
                marginal_2_space_data,
                marginal_2_space_data[i],
                self.bandwidth,
                self.kernel,
            )
            denominator = p_yx_past * p_y_future_y_past
            if denominator <= 0:
                continue

            local_te_values[i] = self._log_base(numerator / denominator)

        # Calculate average TE
        average_te = nanmean(local_te_values)  # Using nanmean to ignore any NaNs

        return average_te, local_te_values
