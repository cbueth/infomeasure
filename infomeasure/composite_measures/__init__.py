"""Composite measures of information."""

from .jsd import jensen_shannon_divergence
from .kld import kullback_leiber_divergence

__all__ = ["jensen_shannon_divergence", "kullback_leiber_divergence"]
