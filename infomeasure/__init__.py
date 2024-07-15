"""infomeasure package."""

# Expose most common functions
from ._version import __version__
from .measures import entropy, mutual_information, transfer_entropy
from .utils import Config

h, mi, te = entropy, mutual_information, transfer_entropy

# Set package attributes
__author__ = "Carlson Büth"

__all__ = ["__version__", "__author__", "estimators"]
