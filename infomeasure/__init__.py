"""infomeasure package."""

# Expose most common functions
from . import estimators
from ._version import __version__

# Set package attributes
__author__ = "Carlson Büth"

__all__ = ["__version__", "__author__", "estimators"]
