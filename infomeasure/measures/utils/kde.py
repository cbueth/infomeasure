"""Kernel Density Estimation (KDE) utilities."""

from numpy import abs as np_abs
from numpy import all as np_all
from numpy import asarray
from numpy import sum as np_sum
from scipy.stats import gaussian_kde


def kde_probability_density_function(data, x, bandwidth, kernel="box"):
    """
    Estimate the probability density function at point x using Kernel Density Estimation
    with a given kernel.


    Parameters
    ----------
    data : array
        A numpy array of data points, where each column represents a dimension.
    x : array-like
        Point at which to estimate the probability density.
    bandwidth : float
        The bandwidth for the kernel.
    kernel : str
        Type of kernel to use ('gaussian' or 'box').

    Returns
    -------
    float
        Estimated probability density at point x.

    Raises
    ------
    ValueError
        If the kernel type is not supported
    ValueError
        If the bandwidth is not a positive number.
    """
    if not isinstance(bandwidth, (float, int)) or bandwidth <= 0:
        raise ValueError("The bandwidth must be a positive number.")
    if kernel == "gaussian":
        kde = gaussian_kde(data.T, bw_method=bandwidth)
        return kde.evaluate(x)
    elif kernel == "box":
        # Convert x to an array to ensure compatibility with numpy operations
        x = asarray(x)
        # Make sure x is 2D for consistent numpy subtraction operation
        # if x.ndim == 1:
        # x = x.reshape((-1, 1))
        # Define the box kernel density estimation
        N, d = data.shape
        # Compute the scaled data by the bandwidth and check if it falls within the unit hypercube centered at x
        scaled_data = np_abs(data - x) / bandwidth
        within_box = np_all(scaled_data <= 0.5, axis=1)
        # Count the number of points inside the box
        count = np_sum(within_box)
        # Normalize by the number of points and the volume of the box (bandwidth^dimension)
        volume = bandwidth**d
        return count / (N * volume)
    else:  # TODO: Add more kernel types as needed
        raise ValueError(f"Unsupported kernel type: {kernel}. Use 'gaussian' or 'box'.")
