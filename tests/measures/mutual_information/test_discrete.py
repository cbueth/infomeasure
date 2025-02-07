"""Explicit discrete mutual information estimator tests."""

import pytest
from numpy import e, log

from infomeasure.estimators.mutual_information import (
    DiscreteMIEstimator,
    DiscreteCMIEstimator,
)


@pytest.mark.parametrize(
    "data_x,data_y,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 10, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 2, 1.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], e, log(2)),
        ([1, 0, 1, 0], [1, 0, 1, 0], 4, log(2) / log(4)),
        ([3, 5, 3, 5, 3, 5], [3, 5, 3, 5, 3, 5], 5, log(2) / log(5)),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 8, log(3) / log(8)),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 2, 2.321928094887362),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 10, 0.6989700043360187),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], "e", 1.6094379124341003),
        ([1, 0, 1, 0], [0, 1, 0, 1], 2, 1.0),
        ([1, 1, 0, 0], [0, 0, 1, 1], 2, 1.0),
        ([1, 1, 0, 0], [0, 1, 0, 1], 2, 0.0),
    ],
)
def test_discrete_mi(data_x, data_y, base, expected):
    """Test the discrete mutual information estimator."""
    est = DiscreteMIEstimator(data_x, data_y, base=base)
    res = est.results()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)


# test with base 2 and different offsets
@pytest.mark.parametrize(
    "data_x,data_y,offset,expected",
    (
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 0, 0.0),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 1, 0.0),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 0, 1.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], 1, 0.918295),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 0, log(3) / log(2)),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 1, 1.521928),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 2, 1.5),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 3, 1.584962),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 4, 1.0),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], 5, 0.0),
        ([2, 4, 4, 3, 1, 3, 1, 4, 4, 2], [4, 4, 2, 4, 2, 1, 3, 4, 3, 2], 0, 0.646439),
        ([2, 4, 4, 3, 1, 3, 1, 4, 4, 2], [4, 4, 2, 4, 2, 1, 3, 4, 3, 2], 2, 1.311278),
        ([2, 4, 4, 3, 1, 3, 1, 4, 4, 2], [4, 4, 2, 4, 2, 1, 3, 4, 3, 2], 5, 1.521928),
        ([2, 4, 4, 3, 1, 3, 1, 4, 4, 2], [4, 4, 2, 4, 2, 1, 3, 4, 3, 2], 6, 1.0),
    ),
)
def test_discrete_mi_offset(data_x, data_y, offset, expected):
    """Test the discrete mutual information estimator with offset."""
    est = DiscreteMIEstimator(data_x, data_y, offset=offset, base=2)
    res = est.results()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)


@pytest.mark.parametrize(
    "data_x,data_y,data_z,base,expected",
    [
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 2, 0.0),
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 10, 0.0),
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0], 2, 0.0),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 1, 5], 2, 0.399999999),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 2, 4, 5], 10, 0.1204119982),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5, 2, 3, 4, 5], "e", 0.2772588722),
        ([1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1], 2, 0.0),
        ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0], 2, 0.688721875),
        ([1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 0, 1], 2, 0.18872187554),
    ],
)
def test_discrete_cmi(data_x, data_y, data_z, base, expected):
    """Test the discrete conditional mutual information estimator."""
    est = DiscreteCMIEstimator(data_x, data_y, data_z, base=base)
    res = est.results()
    assert isinstance(res, float)
    assert res == pytest.approx(expected)
