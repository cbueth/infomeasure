"""Type definitions for the infomeasure package."""

from typing import TypeVar

LogBaseType = int | float | str

EstimatorType = TypeVar("EstimatorType", bound="Estimator")
