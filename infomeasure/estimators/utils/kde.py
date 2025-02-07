"""Kernel Density Estimation (KDE) utilities."""

from numpy import abs as np_abs, newaxis, inf
from numpy import all as np_all
from numpy import asarray
from numpy import sum as np_sum
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde


def kde_probability_density_function(data, bandwidth, x=None, kernel="box"):
    """
    Estimate the probability density function for a given data set using
    Kernel Density Estimation (KDE).

    Parameters
    ----------
    data : array
        A numpy array of data points, where each column represents a dimension.
    bandwidth : float
        The bandwidth for the kernel.
    x : array-like, optional
        Point at which to estimate the probability density.
        If not provided, the function will estimate at all data points.
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

    # If x is not provided, evaluate at all data points
    x = data if x is None else x

    x = asarray(x)
    # Make sure x is 2D for consistent numpy subtraction operation
    if x.ndim == 1:
        x = x[newaxis, :]

    if kernel == "gaussian":
        kde = gaussian_kde(data.T, bw_method=bandwidth)
        return kde.evaluate(x.T).squeeze()
    elif kernel == "box" and x.shape[0] == 1:
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
    elif kernel == "box" and x.shape[0] > 1:
        # Get the number of data points (N) and the number of dimensions (d)
        N, d = data.shape

        # Calculate the volume of the box kernel
        volume = bandwidth**d

        tree = KDTree(data)
        counts = tree.query_ball_point(data, bandwidth / 2, p=inf, return_length=True)
        densities = counts / (N * volume)

        # Squeeze the densities array to remove any single-dimensional entries
        return densities.squeeze()
    # Another approach to box kernel density estimation
    else:  # TODO: Add more kernel types as needed
        raise ValueError(f"Unsupported kernel type: {kernel}. Use 'gaussian' or 'box'.")
