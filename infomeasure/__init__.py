"""infomeasure package."""

# Expose most common functions
from ._version import __version__
from .utils import Config
from .estimators.functional import (
    entropy,
    cross_entropy,
    mutual_information,
    conditional_mutual_information,
    transfer_entropy,
    conditional_transfer_entropy,
    estimator,
    get_estimator_class,
)
from .composite_measures import jensen_shannon_divergence, kullback_leiber_divergence

h, mi, te = entropy, mutual_information, transfer_entropy
hx = cross_entropy
cmi, cte = conditional_mutual_information, conditional_transfer_entropy
jsd, kld = jensen_shannon_divergence, kullback_leiber_divergence

# Set package attributes
__author__ = "Carlson Büth"

__all__ = [
    "__version__",
    "__author__",
    "Config",
    "entropy",
    "cross_entropy",
    "mutual_information",
    "conditional_mutual_information",
    "transfer_entropy",
    "conditional_transfer_entropy",
    "estimator",
    "get_estimator_class",
    "jensen_shannon_divergence",
    "kullback_leiber_divergence",
    # aliases
    "h",
    "hx",
    "mi",
    "cmi",
    "te",
    "cte",
    "jsd",
    "kld",
]
