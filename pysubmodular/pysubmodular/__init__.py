"""
PySubmodular: Python wrapper for SubmodularMinimization.jl

A high-performance Python interface to the Julia SubmodularMinimization.jl package,
providing efficient algorithms for submodular function minimization and general
convex optimization using Wolfe's algorithm.

Main Components:
- SubmodularFunction: Base class for defining submodular functions
- minimize_submodular: Main optimization interface
- wolfe_algorithm: Direct access to Wolfe's algorithm for any polytope
- Built-in functions: ConcaveFunction, CutFunction, FacilityLocation, etc.

Example:
    >>> import pysubmodular as psm
    >>> f = psm.ConcaveFunction(n=10, alpha=0.7)
    >>> result = psm.minimize_submodular(f)
    >>> print(f"Minimum value: {result.min_value}")
"""

from .core import (
    SubmodularFunction,
    OptimizationResult,
    WolfeResult,
    minimize_submodular,
    wolfe_algorithm,
)

from .functions import (
    ConcaveFunction,
    SquareRootFunction,
    MatroidRankFunction,
    CutFunction,
    FacilityLocationFunction,
    WeightedCoverageFunction,
    LogDeterminantFunction,
    EntropyFunction,
)

from .julia_bridge import JuliaBridge
from .exceptions import PySubmodularError, JuliaError, OptimizationError

__version__ = "0.1.0"
__author__ = "Edwin Wlock"
__email__ = "edwlock@example.com"

__all__ = [
    # Core functionality
    "SubmodularFunction",
    "OptimizationResult",
    "WolfeResult",
    "minimize_submodular",
    "wolfe_algorithm",
    
    # Built-in functions
    "ConcaveFunction",
    "SquareRootFunction", 
    "MatroidRankFunction",
    "CutFunction",
    "FacilityLocationFunction",
    "WeightedCoverageFunction",
    "LogDeterminantFunction",
    "EntropyFunction",
    
    # Utilities
    "JuliaBridge",
    
    # Exceptions
    "PySubmodularError",
    "JuliaError", 
    "OptimizationError",
]