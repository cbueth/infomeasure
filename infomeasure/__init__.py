"""infomeasure package."""

# Expose most common functions
from ._version import __version__
from .utils import Config
from .estimators.functional import (
    entropy,
    mutual_information,
    conditional_mutual_information,
    transfer_entropy,
    conditional_transfer_entropy,
    estimator,
)

h, mi, te = entropy, mutual_information, transfer_entropy
cmi, cte = conditional_mutual_information, conditional_transfer_entropy

# Set package attributes
__author__ = "Carlson Büth"

__all__ = [
    "__version__",
    "__author__",
    "Config",
    "entropy",
    "mutual_information",
    "transfer_entropy",
    "h",
    "mi",
    "te",
    "estimator",
]
