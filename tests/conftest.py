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
    params=[
        (getattr(entropy, cls_name), {"bandwidth": 0.3, "kernel": "box"})
        if cls_name == "KernelEntropyEstimator"
        else (getattr(entropy, cls_name), {})
        for cls_name in entropy.__all__
    ],
)
def entropy_estimator(request):
    """A fixture that yields entropy estimator classes, with specific kwargs for one."""
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        (getattr(mutual_information, cls_name), {"bandwidth": 0.3, "kernel": "box"})
        if cls_name == "KernelMIEstimator"
        else (getattr(mutual_information, cls_name), {})
        for cls_name in mutual_information.__all__
    ],
)
def mi_estimator(request):
    """A fixture that yields mutual information estimator classes."""
    return request.param


te_kwargs = {
    "DiscreteTEEstimator": {"k": 4, "l": 4, "delay": 1},
    "KernelTEEstimator": {"bandwidth": 0.3, "kernel": "box"},
}


@pytest.fixture(
    scope="session",
    params=[
        (
            getattr(transfer_entropy, cls_name),
            te_kwargs[cls_name] if cls_name in te_kwargs else {},
        )
        for cls_name in transfer_entropy.__all__
    ],
)
def te_estimator(request):
    """A fixture that yields transfer entropy estimator classes."""
    return request.param
