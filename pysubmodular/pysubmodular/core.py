"""
Core API for PySubmodular.

This module provides the main Python interface for submodular function minimization
and Wolfe's algorithm, with Pythonic design patterns and comprehensive type hints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Optional, Callable, Any, Protocol, runtime_checkable
from typing_extensions import Self

import numpy as np
import numpy.typing as npt

from .julia_bridge import get_julia_bridge
from .exceptions import (
    PySubmodularError, 
    JuliaError, 
    OptimizationError, 
    DimensionError,
    ToleranceError
)


# Type aliases for better readability
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]


@runtime_checkable
class LinearOracle(Protocol):
    """Protocol for linear optimization oracles used by Wolfe's algorithm."""
    
    def __call__(self, c: FloatArray) -> FloatArray:
        """
        Linear optimization oracle.
        
        Args:
            c: Linear objective vector
            
        Returns:
            Vertex of polytope that minimizes dot(c, vertex)
        """
        ...


@dataclass(frozen=True)
class OptimizationResult:
    """
    Result of submodular function minimization.
    
    Attributes:
        minimizer_set: Boolean array indicating the minimizing set
        min_value: Minimum function value achieved
        min_norm_point: Point in base polytope with minimum norm
        iterations: Number of algorithm iterations
        converged: Whether the algorithm converged
        algorithm: Name of algorithm used
    """
    minimizer_set: BoolArray
    min_value: float
    min_norm_point: FloatArray
    iterations: int
    converged: bool
    algorithm: str = "fujishige-wolfe"
    
    @property
    def set_size(self) -> int:
        """Size of the minimizing set."""
        return int(np.sum(self.minimizer_set))
    
    @property
    def selected_indices(self) -> IntArray:
        """Indices of elements in the minimizing set."""
        return np.where(self.minimizer_set)[0]


@dataclass(frozen=True)
class WolfeResult:
    """
    Result of Wolfe's algorithm for minimum norm point computation.
    
    Attributes:
        min_norm_point: Point with minimum norm in the polytope
        norm_value: L2 norm of the minimum norm point
        iterations: Number of algorithm iterations
        converged: Whether the algorithm converged
    """
    min_norm_point: FloatArray
    norm_value: float
    iterations: int
    converged: bool
    
    @property
    def dimension(self) -> int:
        """Dimension of the optimization space."""
        return len(self.min_norm_point)


class SubmodularFunction(ABC):
    """
    Abstract base class for submodular functions.
    
    This class defines the interface that all submodular functions must implement.
    Subclasses should implement the `evaluate` method and set the `n` attribute.
    
    A function f: 2^V -> R is submodular if for all S ⊆ T ⊆ V and v ∉ T:
        f(S ∪ {v}) - f(S) ≥ f(T ∪ {v}) - f(T)
    
    All functions are assumed to be normalized so that f(∅) = 0.
    """
    
    def __init__(self, n: int) -> None:
        """
        Initialize submodular function.
        
        Args:
            n: Size of the ground set
            
        Raises:
            ValueError: If n is not positive
        """
        if n <= 0:
            raise ValueError(f"Ground set size must be positive, got {n}")
        self.n = n
    
    @abstractmethod
    def evaluate(self, subset: BoolArray) -> float:
        """
        Evaluate the function on a given subset.
        
        Args:
            subset: Boolean array indicating which elements are in the subset
            
        Returns:
            Function value for the given subset
            
        Raises:
            DimensionError: If subset has wrong dimension
        """
        pass
    
    def __call__(self, subset: BoolArray) -> float:
        """Make the function callable directly."""
        return self.evaluate(subset)
    
    def _validate_subset(self, subset: BoolArray) -> None:
        """Validate that subset has correct dimensions and type."""
        if not isinstance(subset, np.ndarray):
            raise TypeError("Subset must be a numpy array")
        
        if subset.dtype != bool:
            raise TypeError("Subset must be a boolean array")
        
        if len(subset) != self.n:
            raise DimensionError(self.n, len(subset), "subset evaluation")
    
    def marginal_value(self, subset: BoolArray, element: int) -> float:
        """
        Compute marginal value of adding an element to a subset.
        
        Args:
            subset: Current subset
            element: Element to add (0-indexed)
            
        Returns:
            f(subset ∪ {element}) - f(subset)
        """
        self._validate_subset(subset)
        
        if not (0 <= element < self.n):
            raise ValueError(f"Element index must be in [0, {self.n}), got {element}")
        
        if subset[element]:
            return 0.0  # Element already in subset
        
        # Create subset with element added
        subset_with_element = subset.copy()
        subset_with_element[element] = True
        
        return self.evaluate(subset_with_element) - self.evaluate(subset)
    
    def check_submodularity(self, num_samples: int = 100, tolerance: float = 1e-10) -> bool:
        """
        Check if the function satisfies submodularity (approximately).
        
        Args:
            num_samples: Number of random samples to test
            tolerance: Numerical tolerance for submodularity check
            
        Returns:
            True if function appears to be submodular
        """
        rng = np.random.RandomState(42)  # Reproducible
        
        for _ in range(num_samples):
            # Generate random subsets S ⊆ T
            S = rng.rand(self.n) < 0.3  # Sparse subset
            T = S | (rng.rand(self.n) < 0.4)  # T contains S
            
            # Choose random element not in T
            available = ~T
            if not np.any(available):
                continue
                
            v = rng.choice(np.where(available)[0])
            
            # Check submodularity: f(S∪{v}) - f(S) ≥ f(T∪{v}) - f(T)
            marginal_S = self.marginal_value(S, v)
            marginal_T = self.marginal_value(T, v)
            
            if marginal_S < marginal_T - tolerance:
                return False
        
        return True


def minimize_submodular(
    func: SubmodularFunction,
    tolerance: float = 1e-6,
    max_iterations: int = 10000,
    use_preallocation: bool = False,
    verbose: bool = False
) -> OptimizationResult:
    """
    Minimize a submodular function using the Fujishige-Wolfe algorithm.
    
    This function finds the subset S that minimizes func(S) using the
    polynomial-time Fujishige-Wolfe algorithm based on Wolfe's method
    for finding minimum norm points in polytopes.
    
    Args:
        func: Submodular function to minimize
        tolerance: Convergence tolerance (default: 1e-6)
        max_iterations: Maximum number of iterations (default: 10000)
        use_preallocation: Whether to use pre-allocated workspace for efficiency
        verbose: Whether to print progress information
        
    Returns:
        OptimizationResult containing the minimizer and algorithm information
        
    Raises:
        OptimizationError: If optimization fails
        ToleranceError: If tolerance is invalid
        
    Example:
        >>> import numpy as np
        >>> import pysubmodular as psm
        >>> 
        >>> class SimpleFunction(psm.SubmodularFunction):
        ...     def evaluate(self, subset):
        ...         return np.sum(subset) ** 0.7
        ...
        >>> f = SimpleFunction(5)
        >>> result = psm.minimize_submodular(f)
        >>> print(f"Minimizer: {result.selected_indices}")
        >>> print(f"Min value: {result.min_value:.6f}")
    """
    if tolerance <= 0:
        raise ToleranceError(tolerance, "Tolerance must be positive")
    
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    
    try:
        bridge = get_julia_bridge()
        
        # Create Julia function wrapper
        julia_func = _create_julia_function_wrapper(func, bridge)
        
        # Choose algorithm based on preallocation preference
        if use_preallocation:
            # Create workspace
            workspace = bridge.call_julia_function("WolfeWorkspace", func.n)
            
            # Call pre-allocated version
            result = bridge.call_julia_function(
                "fujishige_wolfe_submodular_minimization!",
                workspace,
                julia_func,
                ε=tolerance,
                max_iterations=max_iterations,
                verbose=verbose
            )
        else:
            # Call automatic version
            result = bridge.call_julia_function(
                "fujishige_wolfe_submodular_minimization",
                julia_func,
                ε=tolerance,
                max_iterations=max_iterations,
                verbose=verbose
            )
        
        # Extract results
        S_min, min_val, x, iterations = result
        
        # Convert to Python types
        minimizer_set = np.array(S_min, dtype=bool)
        min_value = float(min_val)
        min_norm_point = np.array(x, dtype=float)
        num_iterations = int(iterations)
        
        # Check convergence (assume converged if we got a result)
        converged = True
        
        return OptimizationResult(
            minimizer_set=minimizer_set,
            min_value=min_value,
            min_norm_point=min_norm_point,
            iterations=num_iterations,
            converged=converged,
            algorithm="fujishige-wolfe"
        )
        
    except JuliaError as e:
        raise OptimizationError(f"Julia execution failed: {e}", "fujishige-wolfe") from e
    except Exception as e:
        raise OptimizationError(f"Unexpected error: {e}", "fujishige-wolfe") from e


def wolfe_algorithm(
    oracle: LinearOracle,
    dimension: int,
    tolerance: float = 1e-6,
    max_iterations: int = 10000,
    use_preallocation: bool = False,
    verbose: bool = False
) -> WolfeResult:
    """
    Find minimum norm point in a polytope using Wolfe's algorithm.
    
    This is the core algorithm used by minimize_submodular, but can be applied
    to any polytope defined by a linear optimization oracle.
    
    Args:
        oracle: Linear optimization oracle function
        dimension: Dimension of the optimization space
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        use_preallocation: Whether to use pre-allocated workspace
        verbose: Whether to print progress information
        
    Returns:
        WolfeResult containing the minimum norm point and algorithm information
        
    Raises:
        OptimizationError: If optimization fails
        
    Example:
        >>> import numpy as np
        >>> import pysubmodular as psm
        >>> 
        >>> def simplex_oracle(c):
        ...     # Oracle for standard simplex
        ...     i_min = np.argmin(c)
        ...     vertex = np.zeros(len(c))
        ...     vertex[i_min] = 1.0
        ...     return vertex
        ...
        >>> result = psm.wolfe_algorithm(simplex_oracle, dimension=5)
        >>> print(f"Min norm point: {result.min_norm_point}")
        >>> print(f"Norm: {result.norm_value:.6f}")
    """
    if tolerance <= 0:
        raise ToleranceError(tolerance, "Tolerance must be positive")
    
    if dimension <= 0:
        raise ValueError("Dimension must be positive")
    
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    
    try:
        bridge = get_julia_bridge()
        
        # Create Julia oracle wrapper
        julia_oracle = _create_julia_oracle_wrapper(oracle, bridge)
        
        # Choose algorithm based on preallocation preference
        if use_preallocation:
            # Create workspace
            workspace = bridge.call_julia_function("WolfeWorkspace", dimension)
            
            # Call pre-allocated version
            result = bridge.call_julia_function(
                "wolfe_algorithm!",
                workspace,
                julia_oracle,
                ε=tolerance,
                max_iterations=max_iterations,
                verbose=verbose
            )
        else:
            # Call automatic version
            result = bridge.call_julia_function(
                "wolfe_algorithm",
                julia_oracle,
                dimension,
                ε=tolerance,
                max_iterations=max_iterations,
                verbose=verbose
            )
        
        # Extract results
        x, iterations, converged = result
        
        # Convert to Python types
        min_norm_point = np.array(x, dtype=float)
        norm_value = float(np.linalg.norm(min_norm_point))
        num_iterations = int(iterations)
        did_converge = bool(converged)
        
        return WolfeResult(
            min_norm_point=min_norm_point,
            norm_value=norm_value,
            iterations=num_iterations,
            converged=did_converge
        )
        
    except JuliaError as e:
        raise OptimizationError(f"Julia execution failed: {e}", "wolfe") from e
    except Exception as e:
        raise OptimizationError(f"Unexpected error: {e}", "wolfe") from e


def _create_julia_function_wrapper(func: SubmodularFunction, bridge) -> Any:
    """Create a Julia-compatible wrapper for a Python submodular function."""
    try:
        # Check if the function already has a Julia function (built-in functions)
        if hasattr(func, '_julia_func'):
            return func._julia_func
        
        # For custom Python functions, we need to create a Julia wrapper
        # This creates a Julia function that calls back to Python
        wrapper_code = f"""
        struct PythonFunctionWrapper <: SubmodularFunction
            n::Int
            python_id::String
        end
        
        ground_set_size(f::PythonFunctionWrapper) = f.n
        
        function evaluate(f::PythonFunctionWrapper, S::BitVector)
            # This calls back to Python through PyCall
            pyimport("sys").path.insert(0, ".")
            python_bridge = pyimport("pysubmodular.julia_bridge")
            return python_bridge.call_python_function(f.python_id, S)
        end
        """
        
        # Store the Python function reference
        func_id = f"python_func_{id(func)}"
        bridge._main.eval(wrapper_code)
        
        # Store the Python function in the bridge for callback
        if not hasattr(bridge, '_python_functions'):
            bridge._python_functions = {}
        bridge._python_functions[func_id] = func
        
        # Create the Julia wrapper instance
        return bridge.call_julia_function("PythonFunctionWrapper", func.n, func_id)
        
    except Exception as e:
        raise JuliaError(f"Failed to create Julia function wrapper: {e}", e)


def _create_julia_oracle_wrapper(oracle: LinearOracle, bridge) -> Any:
    """Create a Julia-compatible wrapper for a Python linear oracle."""
    try:
        # Create a Julia wrapper for the Python oracle
        wrapper_code = """
        function python_oracle_wrapper(c, oracle_id)
            # This calls back to Python through PyCall
            pyimport("sys").path.insert(0, ".")
            python_bridge = pyimport("pysubmodular.julia_bridge")
            return python_bridge.call_python_oracle(oracle_id, c)
        end
        """
        
        # Store the Python oracle reference
        oracle_id = f"python_oracle_{id(oracle)}"
        bridge._main.eval(wrapper_code)
        
        # Store the Python oracle in the bridge for callback
        if not hasattr(bridge, '_python_oracles'):
            bridge._python_oracles = {}
        bridge._python_oracles[oracle_id] = oracle
        
        # Return a Julia function that can be called with (c) -> oracle(c)
        def julia_oracle(c):
            return bridge._main.eval(f"python_oracle_wrapper({c}, \"{oracle_id}\")")
        
        return julia_oracle
        
    except Exception as e:
        raise JuliaError(f"Failed to create Julia oracle wrapper: {e}", e)