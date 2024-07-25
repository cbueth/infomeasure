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
        The source and destination data used to estimate the transfer entropy.
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
        numerator_term1, numerator_term2, denominator_term1, denominator_term2 = (
            te_observations(
                self.source, self.dest, self.src_hist_len, self.dest_hist_len
            )
        )
        local_te_values = zeros(len(numerator_term1))

        # Compute KDE for each term directly using slices
        for i in range(len(numerator_term1)):
            # g(x_{i+1}, x_i^{(k)}, y_i^{(l)})
            p_x_future_x_past_y_past = kde_probability_density_function(
                numerator_term1, numerator_term1[i], self.bandwidth, self.kernel
            )
            if p_x_future_x_past_y_past == 0:
                continue
            # g(x_i^{(k)})
            p_x_past = kde_probability_density_function(
                numerator_term2, numerator_term2[i], self.bandwidth, self.kernel
            )
            numerator = p_x_future_x_past_y_past * p_x_past
            if numerator <= 0:
                continue
            # g(x_i^{(k)}, y_i^{(l)})
            p_xy_past = kde_probability_density_function(
                denominator_term1, denominator_term1[i], self.bandwidth, self.kernel
            )
            if p_xy_past == 0:
                continue
            # g(x_{i+1}, x_i^{(k)})
            p_x_future_x_past = kde_probability_density_function(
                denominator_term2, denominator_term2[i], self.bandwidth, self.kernel
            )
            denominator = p_xy_past * p_x_future_x_past
            if denominator <= 0:
                continue

            local_te_values[i] = self._log_base(numerator / denominator)

        # Calculate average TE
        average_te = nanmean(local_te_values)  # Using nanmean to ignore any NaNs

        return average_te, local_te_values
