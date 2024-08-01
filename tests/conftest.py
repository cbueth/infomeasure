"""Module for test fixtures available for all test files"""

import pytest
from numpy.random import default_rng as rng

from infomeasure import Config
from infomeasure.measures import entropy, mutual_information, transfer_entropy


@pytest.fixture(
    scope="function",
)
def default_rng():
    """A random number generator."""
    return rng(seed=798)


@pytest.fixture(autouse=True, scope="session")
def activate_debug_logging():
    """Activate debug logging for all tests."""
    Config.set_log_level("DEBUG")


@pytest.fixture(
    scope="session",
    params=entropy.__all__,
)
def entropy_estimator(request):
    """A fixture that yields entropy estimator classes, with specific kwargs for one."""
    kwargs = {
        "KernelEntropyEstimator": {"bandwidth": 0.3, "kernel": "box"},
        "SymbolicEntropyEstimator": {"order": 2},
        "RenyiEntropyEstimator": {"alpha": 1.5},
    }
    return getattr(entropy, request.param), kwargs.get(request.param, {})


@pytest.fixture(
    scope="session",
    params=mutual_information.__all__,
)
def mi_estimator(request):
    """A fixture that yields mutual information estimator classes."""
    kwargs = {
        "KernelMIEstimator": {"bandwidth": 0.3, "kernel": "box"},
        "SymbolicMIEstimator": {"order": 2},
        "RenyiMIEstimator": {"alpha": 1.5},
    }
    return getattr(mutual_information, request.param), kwargs.get(request.param, {})


@pytest.fixture(
    scope="session",
    params=transfer_entropy.__all__,
)
def te_estimator(request):
    """A fixture that yields transfer entropy estimator classes."""
    kwargs = {
        "KernelTEEstimator": {"bandwidth": 0.3, "kernel": "box"},
        "SymbolicTEEstimator": {"order": 2},
    }
    return getattr(transfer_entropy, request.param), kwargs.get(request.param, {})
