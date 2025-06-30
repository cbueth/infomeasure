"""Custom exceptions for the infomeasure package."""


class TheoreticalInconsistencyError(Exception):
    """Exception raised when a method cannot be implemented due to theoretical inconsistencies.

    This exception is used for cases where the mathematical or theoretical foundation
    of a method makes it inappropriate or impossible to implement in a meaningful way.
    Examples include cross-entropy for estimators that mix bias corrections from
    different distributions, or methods that violate fundamental assumptions.

    This is different from NotImplementedError, which indicates that implementation
    is planned but not yet done. TheoreticalInconsistencyError indicates that
    implementation is not theoretically sound or appropriate.
    """

    pass
