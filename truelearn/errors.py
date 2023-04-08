__all__ = [
    "TrueLearnError",
    "TrueLearnTypeError",
    "TrueLearnValueError",
    "InvalidArgumentError",
    "WikifierError",
]


class TrueLearnError(Exception):
    """Base-class for all exceptions raised by truelearn library."""


class TrueLearnTypeError(TrueLearnError, TypeError):
    """Base-class for all type-related exceptions."""


class TrueLearnValueError(TrueLearnError, ValueError):
    """Base-class for all value-related exceptions."""


class InvalidArgumentError(TrueLearnError):
    """Invalid argument.

    This is raised when an unexpected argument is given to the method/class.
    """


class WikifierError(TrueLearnError):
    """Errors related to Wikifier."""
