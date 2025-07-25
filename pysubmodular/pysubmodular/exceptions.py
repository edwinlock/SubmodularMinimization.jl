"""
Custom exceptions for PySubmodular.

This module defines custom exception classes that provide clear error messages
and proper error handling for the Python-Julia interface.
"""

from typing import Optional, Any


class PySubmodularError(Exception):
    """Base exception class for PySubmodular."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.cause = cause


class JuliaError(PySubmodularError):
    """Exception raised when Julia code execution fails."""
    
    def __init__(self, message: str, julia_exception: Optional[Any] = None) -> None:
        super().__init__(f"Julia error: {message}")
        self.julia_exception = julia_exception


class OptimizationError(PySubmodularError):
    """Exception raised when optimization fails."""
    
    def __init__(self, message: str, algorithm: str = "unknown") -> None:
        super().__init__(f"Optimization failed ({algorithm}): {message}")
        self.algorithm = algorithm


class SubmodularityError(PySubmodularError):
    """Exception raised when a function violates submodularity."""
    
    def __init__(self, message: str = "Function does not satisfy submodularity property") -> None:
        super().__init__(message)


class DimensionError(PySubmodularError):
    """Exception raised when array dimensions are incompatible."""
    
    def __init__(self, expected: int, got: int, context: str = "") -> None:
        message = f"Expected dimension {expected}, got {got}"
        if context:
            message += f" in {context}"
        super().__init__(message)
        self.expected = expected
        self.got = got


class ToleranceError(PySubmodularError):
    """Exception raised when tolerance parameters are invalid."""
    
    def __init__(self, tolerance: float, message: str = "Invalid tolerance") -> None:
        super().__init__(f"{message}: {tolerance}")
        self.tolerance = tolerance