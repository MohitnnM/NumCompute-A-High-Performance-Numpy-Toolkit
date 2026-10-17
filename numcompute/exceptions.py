"""
Custom exceptions shared across the NumCompute package.

Kept in their own module so any module (and users) can import them
without pulling in helper functions or causing circular imports.
"""


class NotFittedError(ValueError, AttributeError):
    """
        Raised when transform or predict is called before fit.

        Inherits from ValueError and AttributeError, so code that catches
        either built-in exception will still catch this one.
    """
