"""Helper function for the unit ball volume."""

from numpy import pi
from scipy.special import gamma


def unit_ball_volume(d, r=1, p=2):
    r"""Calculate the volume of the d-dimensional ball with radius r in :math:`L^p` norm.

    .. math::

        V_d = \begin{cases}
            2r & \text{if } d = 1 \text{ and } p = 2, \\
            \frac{4\pi r^3}{3} & \text{if } d = 3 \text{ and } p = 2, \\
            (2r)^d & \text{if } p = \infty, \\
            \frac{(\pi r^2)^{d/2}}{\Gamma(1 + d/2)} & \text{if } p = 2, \\
            \frac{(2r)^d \Gamma(1 + \frac{1}{p})^d}{\Gamma(1 + \frac{d}{p})} & \text{otherwise}.
        \end{cases}

    Parameters
    ----------
    d : int
        The dimensionality of the space.
    p : float
        The :math:`L^p` norm.
    r : float, optional
        The radius of the ball (default is 1).

    Returns
    -------
    float
        The volume of the d-dimensional ball with radius r in :math:`L^p` norm.
    """
    if p == float("inf"):
        return (2 * r) ** d
    elif p == 2:
        if d == 1:
            return 2 * r
        if d == 2:
            return pi * r**2
        elif d == 3:
            return (4 / 3) * pi * r**3
        else:
            return (pi ** (d / 2) * r**d) / gamma(1 + d / 2)
    else:
        return ((2 * r) ** d * gamma(1 + 1 / p) ** d) / gamma(1 + d / p)
