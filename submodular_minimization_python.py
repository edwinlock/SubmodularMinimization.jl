"""
SubmodularMinimization Python Wrapper

A Python interface to the SubmodularMinimization.jl library using ctypes.
This wrapper provides a clean, object-oriented API for submodular function
minimization with function objects.

Requirements:
    - The Julia library must be built using build_python_library.jl
    - The library path must be set in environment variables

Usage:
    from submodular_minimization_python import SubmodularMinimizer
    
    # Create minimizer instance
    solver = SubmodularMinimizer("python_lib/lib/libsubmodular")
    
    # Create function objects
    f1 = solver.create_concave_function(n=10, alpha=0.7)
    f2 = solver.create_cut_function(n=4, edges=[(0,1), (1,2), (2,3), (0,3)])
    
    # Solve with unified interface
    result = solver.solve(f1)
    print(f"Optimal set: {result.optimal_set}")
    print(f"Minimum value: {result.min_value}")
    
    # Use with other algorithms
    wolfe_result = solver.wolfe_algorithm(f1)
    is_submodular = solver.check_submodular(f1)
"""

import ctypes
import os
import platform
from typing import List, Tuple, Optional, NamedTuple, Callable
from abc import ABC, abstractmethod
import numpy as np


class SubmodularFunction(ABC):
    """
    Abstract base class for submodular functions.
    
    All submodular function objects inherit from this class and provide
    the necessary information for the C interface.
    """
    
    def __init__(self, n: int):
        """Initialize with ground set size."""
        self.n = n
    
    @property
    @abstractmethod
    def func_type(self) -> int:
        """Return the C function type constant."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> List[float]:
        """Return the parameter array for the C interface."""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the function."""
        pass


class ConcaveFunction(SubmodularFunction):
    """Concave submodular function f(S) = |S|^alpha."""
    
    def __init__(self, n: int, alpha: float):
        super().__init__(n)
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1 for submodularity")
        self.alpha = alpha
    
    @property
    def func_type(self) -> int:
        return 1  # FUNC_TYPE_CONCAVE
    
    @property
    def parameters(self) -> List[float]:
        return [self.alpha]
    
    def __repr__(self) -> str:
        return f"ConcaveFunction(n={self.n}, alpha={self.alpha})"


class CutFunction(SubmodularFunction):
    """Graph cut function."""
    
    def __init__(self, n: int, edges: List[Tuple[int, int]]):
        super().__init__(n)
        self.edges = edges
        # Validate edges
        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"Edge ({u}, {v}) is out of bounds for n={n}")
    
    @property
    def func_type(self) -> int:
        return 2  # FUNC_TYPE_CUT
    
    @property
    def parameters(self) -> List[float]:
        # Flatten edges into parameter array (convert to 1-indexed for Julia)
        params = []
        for u, v in self.edges:
            params.extend([float(u + 1), float(v + 1)])
        return params
    
    def __repr__(self) -> str:
        return f"CutFunction(n={self.n}, edges={self.edges})"


class SquareRootFunction(SubmodularFunction):
    """Square root function f(S) = sqrt(|S|)."""
    
    def __init__(self, n: int):
        super().__init__(n)
    
    @property
    def func_type(self) -> int:
        return 3  # FUNC_TYPE_SQRT
    
    @property
    def parameters(self) -> List[float]:
        return []  # No parameters needed
    
    def __repr__(self) -> str:
        return f"SquareRootFunction(n={self.n})"


class MatroidFunction(SubmodularFunction):
    """Matroid rank function f(S) = min(|S|, k)."""
    
    def __init__(self, n: int, k: int):
        super().__init__(n)
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k = k
    
    @property
    def func_type(self) -> int:
        return 4  # FUNC_TYPE_MATROID
    
    @property
    def parameters(self) -> List[float]:
        return [float(self.k)]
    
    def __repr__(self) -> str:
        return f"MatroidFunction(n={self.n}, k={self.k})"


class BipartiteMatchingFunction(SubmodularFunction):
    """Bipartite matching function."""
    
    def __init__(self, n1: int, n2: int, density: float):
        super().__init__(n1 + n2)  # Total vertices
        self.n1 = n1
        self.n2 = n2
        self.density = density
        if not (0 <= density <= 1):
            raise ValueError("Density must be between 0 and 1")
    
    @property
    def func_type(self) -> int:
        return 5  # FUNC_TYPE_BIPARTITE_MATCHING
    
    @property
    def parameters(self) -> List[float]:
        return [float(self.n1), float(self.n2), self.density]
    
    def __repr__(self) -> str:
        return f"BipartiteMatchingFunction(n1={self.n1}, n2={self.n2}, density={self.density})"


class FacilityLocationFunction(SubmodularFunction):
    """Facility location function."""
    
    def __init__(self, num_facilities: int, num_clients: int, weights: Optional[np.ndarray] = None):
        super().__init__(num_clients)  # Ground set is clients
        self.num_facilities = num_facilities
        self.num_clients = num_clients
        self.weights = weights
    
    @property
    def func_type(self) -> int:
        return 6  # FUNC_TYPE_FACILITY_LOCATION
    
    @property
    def parameters(self) -> List[float]:
        params = [float(self.num_facilities), float(self.num_clients)]
        if self.weights is not None:
            # Flatten weights matrix
            params.extend(self.weights.flatten().tolist())
        return params
    
    def __repr__(self) -> str:
        weights_info = "custom weights" if self.weights is not None else "random weights"
        return f"FacilityLocationFunction(facilities={self.num_facilities}, clients={self.num_clients}, {weights_info})"


class CustomFunction(SubmodularFunction):
    """
    Custom submodular function defined by evaluation table.
    
    For small n, you can provide all 2^n function evaluations.
    """
    
    def __init__(self, evaluations: List[float]):
        n = int(np.log2(len(evaluations)))
        if 2**n != len(evaluations):
            raise ValueError(f"Evaluations length {len(evaluations)} is not a power of 2")
        super().__init__(n)
        self.evaluations = evaluations
    
    @property
    def func_type(self) -> int:
        # We'll need to add a custom function type to the Julia interface
        raise NotImplementedError("Custom functions not yet implemented in C interface")
    
    @property
    def parameters(self) -> List[float]:
        return self.evaluations
    
    def __repr__(self) -> str:
        return f"CustomFunction(n={self.n}, evaluations={len(self.evaluations)} values)"


class CallbackFunction(SubmodularFunction):
    """
    Submodular function defined by a Python callback.
    
    The callback function should take a list of indices (0-based) and return a float.
    """
    
    def __init__(self, callback: Callable[[List[int]], float], n: int):
        super().__init__(n)
        self.callback = callback
        self._callback_wrapper = None
        
    @property
    def func_type(self) -> int:
        return 7  # FUNC_TYPE_CALLBACK
    
    @property
    def parameters(self) -> List[float]:
        return [float(self.n)]  # Only need to pass the ground set size
    
    def get_callback_wrapper(self):
        """Get ctypes callback wrapper for the Python function."""
        if self._callback_wrapper is None:
            # Define the C callback signature: double callback(int* indices, int n_indices)
            CALLBACK_TYPE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
            
            def c_callback(indices_ptr, n_indices):
                try:
                    # Convert C array to Python list
                    if n_indices > 0:
                        indices = [indices_ptr[i] for i in range(n_indices)]
                    else:
                        indices = []
                    
                    # Call the Python function
                    result = self.callback(indices)
                    return float(result)
                except Exception as e:
                    print(f"Error in Python callback: {e}")
                    return float('nan')
            
            self._callback_wrapper = CALLBACK_TYPE(c_callback)
        
        return self._callback_wrapper
    
    def __repr__(self) -> str:
        return f"CallbackFunction(n={self.n}, callback={self.callback.__name__ if hasattr(self.callback, '__name__') else 'lambda'})"


class SubmodularResult(NamedTuple):
    """Result from submodular minimization."""
    optimal_set: List[int]
    min_value: float
    iterations: int
    success: bool
    error_message: str = ""

class WolfeResult(NamedTuple):
    """Result from Wolfe algorithm (minimum norm point)."""
    min_norm_point: List[float]
    iterations: int
    converged: bool
    success: bool
    error_message: str = ""

class SubmodularChecker(NamedTuple):
    """Result from submodularity check."""
    is_submodular: bool
    violations: int
    success: bool
    error_message: str = ""

class OptimalityCheck(NamedTuple):
    """Result from optimality verification."""
    is_optimal: bool
    improvement_value: float
    success: bool
    error_message: str = ""

class SubmodularMinimizer:
    """
    Python interface to SubmodularMinimization.jl library.
    
    This class provides access to high-performance submodular function
    minimization algorithms implemented in Julia, using a clean function
    object-based API.
    """
    
    # Error codes
    SUCCESS = 0
    ERROR_INVALID_FUNCTION_TYPE = -1
    ERROR_INVALID_PARAMETERS = -2
    ERROR_CONVERGENCE_FAILED = -3
    ERROR_MEMORY_ALLOCATION = -4
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the SubmodularMinimizer.
        
        Args:
            library_path: Path to the compiled Julia library.
                         If None, tries to find it in common locations.
        """
        self.lib = None
        self._load_library(library_path)
        self._setup_function_signatures()
        
    def _load_library(self, library_path: Optional[str]):
        """Load the dynamic library."""
        if library_path is None:
            # Try common locations
            candidates = [
                "python_lib/lib/libsubmodular.so",
                "python_lib/lib/libsubmodular.dylib", 
                "python_lib/lib/libsubmodular.dll",
                "libsubmodular.so",
                "libsubmodular.dylib",
                "libsubmodular.dll"
            ]
            
            for candidate in candidates:
                if os.path.exists(candidate):
                    library_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    "Could not find SubmodularMinimization library. "
                    "Please build it first with 'julia build_python_library.jl' "
                    "or specify the path manually."
                )
        
        try:
            self.lib = ctypes.CDLL(library_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load library {library_path}: {e}")
            
    def _setup_function_signatures(self):
        """Setup ctypes function signatures."""
        # fujishige_wolfe_solve_c
        self.lib.fujishige_wolfe_solve_c.argtypes = [
            ctypes.c_int32,  # func_type
            ctypes.POINTER(ctypes.c_double),  # params
            ctypes.c_int32,  # n_params
            ctypes.c_int32,  # n
            ctypes.c_double,  # tolerance
            ctypes.c_int32,  # max_iterations
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),  # result_set
            ctypes.POINTER(ctypes.c_double),  # result_value
            ctypes.POINTER(ctypes.c_int32),  # result_iterations
        ]
        self.lib.fujishige_wolfe_solve_c.restype = ctypes.c_int32
        
        # wolfe_algorithm_c
        self.lib.wolfe_algorithm_c.argtypes = [
            ctypes.c_int32,  # func_type
            ctypes.POINTER(ctypes.c_double),  # params
            ctypes.c_int32,  # n_params
            ctypes.c_int32,  # n
            ctypes.c_double,  # tolerance
            ctypes.c_int32,  # max_iterations
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # result_x
            ctypes.POINTER(ctypes.c_int32),  # result_iterations
            ctypes.POINTER(ctypes.c_int32),  # converged
        ]
        self.lib.wolfe_algorithm_c.restype = ctypes.c_int32
        
        # check_submodular_c
        self.lib.check_submodular_c.argtypes = [
            ctypes.c_int32,  # func_type
            ctypes.POINTER(ctypes.c_double),  # params
            ctypes.c_int32,  # n_params
            ctypes.c_int32,  # n
            ctypes.POINTER(ctypes.c_int32),  # violations
        ]
        self.lib.check_submodular_c.restype = ctypes.c_int32
        
        # is_minimiser_c
        self.lib.is_minimiser_c.argtypes = [
            ctypes.c_int32,  # func_type
            ctypes.POINTER(ctypes.c_double),  # params
            ctypes.c_int32,  # n_params
            ctypes.c_int32,  # n
            ctypes.POINTER(ctypes.c_int32),  # candidate_set
            ctypes.c_int32,  # set_size
            ctypes.POINTER(ctypes.c_double),  # improvement_value
        ]
        self.lib.is_minimiser_c.restype = ctypes.c_int32
        
        # register_python_callback
        self.lib.register_python_callback.argtypes = [ctypes.c_void_p]
        self.lib.register_python_callback.restype = ctypes.c_int32
        
        # test_python_callback
        self.lib.test_python_callback.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double)
        ]
        self.lib.test_python_callback.restype = ctypes.c_int32
        
        # cache management functions
        self.lib.clear_memoization_cache.argtypes = []
        self.lib.clear_memoization_cache.restype = ctypes.c_int32
        
        self.lib.get_cache_stats.argtypes = [
            ctypes.POINTER(ctypes.c_int32),  # cache_hits
            ctypes.POINTER(ctypes.c_int32),  # cache_misses  
            ctypes.POINTER(ctypes.c_int32),  # cache_size
        ]
        self.lib.get_cache_stats.restype = ctypes.c_int32
        
        self.lib.set_cache_enabled.argtypes = [ctypes.c_int32]
        self.lib.set_cache_enabled.restype = ctypes.c_int32
        
    # Factory methods for creating function objects
    def create_concave_function(self, n: int, alpha: float) -> ConcaveFunction:
        """Create a concave submodular function f(S) = |S|^alpha."""
        return ConcaveFunction(n, alpha)
    
    def create_cut_function(self, n: int, edges: List[Tuple[int, int]]) -> CutFunction:
        """Create a graph cut function."""
        return CutFunction(n, edges)
    
    def create_sqrt_function(self, n: int) -> SquareRootFunction:
        """Create a square root function f(S) = sqrt(|S|)."""
        return SquareRootFunction(n)
    
    def create_matroid_function(self, n: int, k: int) -> MatroidFunction:
        """Create a matroid rank function f(S) = min(|S|, k)."""
        return MatroidFunction(n, k)
    
    def create_bipartite_matching_function(self, n1: int, n2: int, density: float) -> BipartiteMatchingFunction:
        """Create a bipartite matching function."""
        return BipartiteMatchingFunction(n1, n2, density)
    
    def create_facility_location_function(self, num_facilities: int, num_clients: int, 
                                        weights: Optional[np.ndarray] = None) -> FacilityLocationFunction:
        """Create a facility location function."""
        return FacilityLocationFunction(num_facilities, num_clients, weights)
    
    def create_custom_function(self, evaluations: List[float]) -> CustomFunction:
        """Create a custom function from evaluation table (for small n)."""
        return CustomFunction(evaluations)
    
    def create_callback_function(self, callback: Callable[[List[int]], float], n: int) -> CallbackFunction:
        """Create a submodular function from a Python callback."""
        return CallbackFunction(callback, n)
    
    def _register_callback(self, callback_func: CallbackFunction):
        """Register a callback function with Julia."""
        wrapper = callback_func.get_callback_wrapper()
        callback_ptr = ctypes.cast(wrapper, ctypes.c_void_p)
        
        result = self.lib.register_python_callback(callback_ptr)
        if result != self.SUCCESS:
            raise RuntimeError(f"Failed to register Python callback: error code {result}")
            
        # Keep a reference to prevent garbage collection
        if not hasattr(self, '_registered_callbacks'):
            self._registered_callbacks = []
        self._registered_callbacks.append(wrapper)
    
    def clear_cache(self):
        """Clear the memoization cache for callback functions."""
        result = self.lib.clear_memoization_cache()
        if result != self.SUCCESS:
            raise RuntimeError(f"Failed to clear cache: error code {result}")
    
    def get_cache_stats(self):
        """Get cache statistics for callback functions."""
        hits = ctypes.c_int32()
        misses = ctypes.c_int32()
        size = ctypes.c_int32()
        
        result = self.lib.get_cache_stats(
            ctypes.byref(hits),
            ctypes.byref(misses), 
            ctypes.byref(size)
        )
        
        if result != self.SUCCESS:
            raise RuntimeError(f"Failed to get cache stats: error code {result}")
        
        total_calls = hits.value + misses.value
        hit_rate = hits.value / total_calls if total_calls > 0 else 0.0
        
        return {
            'cache_hits': hits.value,
            'cache_misses': misses.value,
            'cache_size': size.value,
            'total_calls': total_calls,
            'hit_rate': hit_rate
        }
    
    # Unified solving interface
    def solve(self, func: SubmodularFunction, tolerance: float = 1e-6, 
              max_iterations: int = 10000) -> SubmodularResult:
        """
        Solve a submodular minimization problem using Fujishige-Wolfe algorithm.
        
        Args:
            func: SubmodularFunction object
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            SubmodularResult with optimal set and metadata
        """
        # Special handling for callback functions
        if isinstance(func, CallbackFunction):
            self._register_callback(func)
        
        # Extract function information
        params = func.parameters
        param_array = (ctypes.c_double * len(params))(*params)
        result_set_ptr = ctypes.POINTER(ctypes.c_int32)()
        result_value = ctypes.c_double()
        result_iterations = ctypes.c_int32()
        
        # Call Julia function
        set_size = self.lib.fujishige_wolfe_solve_c(
            func.func_type,
            param_array,
            len(params),
            func.n,
            tolerance,
            max_iterations,
            ctypes.byref(result_set_ptr),
            ctypes.byref(result_value),
            ctypes.byref(result_iterations)
        )
        
        # Check for errors
        if set_size < 0:
            error_messages = {
                self.ERROR_INVALID_FUNCTION_TYPE: "Invalid function type",
                self.ERROR_INVALID_PARAMETERS: "Invalid parameters", 
                self.ERROR_CONVERGENCE_FAILED: "Convergence failed",
                self.ERROR_MEMORY_ALLOCATION: "Memory allocation error"
            }
            return SubmodularResult(
                optimal_set=[],
                min_value=float('nan'),
                iterations=0,
                success=False,
                error_message=error_messages.get(set_size, f"Unknown error code: {set_size}")
            )
        
        # Extract optimal set
        if set_size > 0 and result_set_ptr:
            optimal_set = [result_set_ptr[i] for i in range(set_size)]
        else:
            optimal_set = []
            
        return SubmodularResult(
            optimal_set=optimal_set,
            min_value=result_value.value,
            iterations=result_iterations.value,
            success=True
        )
    
    def wolfe_algorithm(self, func: SubmodularFunction, tolerance: float = 1e-6, 
                       max_iterations: int = 10000) -> WolfeResult:
        """
        Run Wolfe algorithm to find minimum norm point in base polytope.
        
        Args:
            func: SubmodularFunction object
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            WolfeResult with minimum norm point and metadata
        """
        # Special handling for callback functions
        if isinstance(func, CallbackFunction):
            self._register_callback(func)
            
        # Extract function information
        params = func.parameters
        param_array = (ctypes.c_double * len(params))(*params)
        result_x_ptr = ctypes.POINTER(ctypes.c_double)()
        result_iterations = ctypes.c_int32()
        converged = ctypes.c_int32()
        
        # Call Julia function
        result = self.lib.wolfe_algorithm_c(
            func.func_type,
            param_array,
            len(params),
            func.n,
            tolerance,
            max_iterations,
            ctypes.byref(result_x_ptr),
            ctypes.byref(result_iterations),
            ctypes.byref(converged)
        )
        
        # Check for errors
        if result != self.SUCCESS:
            error_messages = {
                self.ERROR_INVALID_FUNCTION_TYPE: "Invalid function type",
                self.ERROR_INVALID_PARAMETERS: "Invalid parameters",
                self.ERROR_CONVERGENCE_FAILED: "Convergence failed",
                self.ERROR_MEMORY_ALLOCATION: "Memory allocation error"
            }
            return WolfeResult(
                min_norm_point=[],
                iterations=0,
                converged=False,
                success=False,
                error_message=error_messages.get(result, f"Unknown error code: {result}")
            )
        
        # Extract minimum norm point
        if result_x_ptr:
            min_norm_point = [result_x_ptr[i] for i in range(func.n)]
        else:
            min_norm_point = []
            
        return WolfeResult(
            min_norm_point=min_norm_point,
            iterations=result_iterations.value,
            converged=(converged.value == 1),
            success=True
        )
    
    def check_submodular(self, func: SubmodularFunction) -> SubmodularChecker:
        """
        Check if a function is submodular.
        
        Args:
            func: SubmodularFunction object
            
        Returns:
            SubmodularChecker with results
        """
        # Special handling for callback functions
        if isinstance(func, CallbackFunction):
            self._register_callback(func)
            
        params = func.parameters
        param_array = (ctypes.c_double * len(params))(*params)
        violations = ctypes.c_int32()
        
        result = self.lib.check_submodular_c(
            func.func_type,
            param_array,
            len(params),
            func.n,
            ctypes.byref(violations)
        )
        
        if result < 0:
            error_messages = {
                self.ERROR_INVALID_FUNCTION_TYPE: "Invalid function type",
                self.ERROR_INVALID_PARAMETERS: "Invalid parameters"
            }
            return SubmodularChecker(
                is_submodular=False,
                violations=0,
                success=False,
                error_message=error_messages.get(result, f"Unknown error: {result}")
            )
        
        return SubmodularChecker(
            is_submodular=(result == 1),
            violations=violations.value,
            success=True
        )
    
    def check_optimality(self, func: SubmodularFunction, candidate_set: List[int]) -> OptimalityCheck:
        """
        Check if a given set is optimal.
        
        Args:
            func: SubmodularFunction object
            candidate_set: Set to check (0-indexed)
            
        Returns:
            OptimalityCheck with results
        """
        # Special handling for callback functions
        if isinstance(func, CallbackFunction):
            self._register_callback(func)
            
        params = func.parameters
        param_array = (ctypes.c_double * len(params))(*params)
        set_array = (ctypes.c_int32 * len(candidate_set))(*candidate_set)
        improvement_value = ctypes.c_double()
        
        result = self.lib.is_minimiser_c(
            func.func_type,
            param_array,
            len(params),
            func.n,
            set_array,
            len(candidate_set),
            ctypes.byref(improvement_value)
        )
        
        if result < 0:
            return OptimalityCheck(
                is_optimal=False,
                improvement_value=float('nan'),
                success=False,
                error_message=f"Error code: {result}"
            )
        
        return OptimalityCheck(
            is_optimal=(result == 1),
            improvement_value=improvement_value.value,
            success=True
        )


def demo():
    """Demonstrate the Python wrapper with function objects."""
    print("SubmodularMinimization.jl Python Demo (Function Objects)")
    print("=" * 55)
    
    try:
        # Initialize solver
        solver = SubmodularMinimizer()
        print("✅ Library loaded successfully!")
        
        # Test concave function
        print("\n🧮 Testing concave function f(S) = |S|^0.7")
        f1 = solver.create_concave_function(n=8, alpha=0.7)
        print(f"  Function: {f1}")
        
        result = solver.solve(f1)
        if result.success:
            print(f"  Optimal set: {result.optimal_set}")
            print(f"  Minimum value: {result.min_value:.6f}")
            print(f"  Iterations: {result.iterations}")
            
            # Check submodularity
            check = solver.check_submodular(f1)
            if check.success:
                print(f"  Is submodular: {check.is_submodular}")
                if not check.is_submodular:
                    print(f"  Violations: {check.violations}")
        else:
            print(f"  ❌ Error: {result.error_message}")
        
        # Test cut function
        print("\n📊 Testing cut function on small graph")
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]  # Square graph
        f2 = solver.create_cut_function(n=4, edges=edges)
        print(f"  Function: {f2}")
        
        result = solver.solve(f2)
        if result.success:
            print(f"  Optimal cut: {result.optimal_set}")
            print(f"  Cut value: {result.min_value:.6f}")
            print(f"  Iterations: {result.iterations}")
        else:
            print(f"  ❌ Error: {result.error_message}")
            
        # Test square root function
        print("\n√ Testing square root function f(S) = √|S|")
        f3 = solver.create_sqrt_function(n=6)
        print(f"  Function: {f3}")
        
        result = solver.solve(f3)
        if result.success:
            print(f"  Optimal set: {result.optimal_set}")
            print(f"  Minimum value: {result.min_value:.6f}")
            print(f"  Iterations: {result.iterations}")
        else:
            print(f"  ❌ Error: {result.error_message}")
            
        # Test Wolfe algorithm directly
        print("\n🎯 Testing Wolfe algorithm on concave function")
        f4 = solver.create_concave_function(n=6, alpha=0.7)
        wolfe_result = solver.wolfe_algorithm(f4)
        
        if wolfe_result.success:
            print(f"  Minimum norm point: {[f'{x:.4f}' for x in wolfe_result.min_norm_point]}")
            print(f"  Converged: {wolfe_result.converged}")
            print(f"  Iterations: {wolfe_result.iterations}")
        else:
            print(f"  ❌ Error: {wolfe_result.error_message}")
            
        # Test matroid function
        print("\n🔧 Testing matroid function f(S) = min(|S|, 3)")
        f5 = solver.create_matroid_function(n=8, k=3)
        print(f"  Function: {f5}")
        
        result = solver.solve(f5)
        if result.success:
            print(f"  Optimal set: {result.optimal_set}")
            print(f"  Minimum value: {result.min_value:.6f}")
            print(f"  Iterations: {result.iterations}")
            
            # Check optimality
            opt_check = solver.check_optimality(f5, result.optimal_set)
            if opt_check.success:
                print(f"  Solution is optimal: {opt_check.is_optimal}")
        else:
            print(f"  ❌ Error: {result.error_message}")
            
        # Test Python callback function
        print("\n🐍 Testing Python callback function")
        def my_custom_function(subset_indices):
            """Custom submodular function: f(S) = |S|^0.6 + penalty for large sets"""
            size = len(subset_indices)
            base_value = size ** 0.6
            penalty = max(0, size - 5) * 0.5  # Penalty for sets larger than 5
            return base_value + penalty
        
        f6 = solver.create_callback_function(my_custom_function, n=8)
        print(f"  Function: {f6}")
        
        # Clear cache before solving
        solver.clear_cache()
        
        result = solver.solve(f6)
        if result.success:
            print(f"  Optimal set: {result.optimal_set}")
            print(f"  Minimum value: {result.min_value:.6f}")
            print(f"  Iterations: {result.iterations}")
            
            # Show cache performance
            cache_stats = solver.get_cache_stats()
            print(f"  Cache performance:")
            print(f"    Total function calls: {cache_stats['total_calls']}")
            print(f"    Cache hits: {cache_stats['cache_hits']}")
            print(f"    Cache misses: {cache_stats['cache_misses']}")
            print(f"    Hit rate: {cache_stats['hit_rate']:.1%}")
            print(f"    Cache size: {cache_stats['cache_size']} entries")
            
            # Verify the result by manually calling the function
            manual_value = my_custom_function(result.optimal_set)
            print(f"  Manual verification: f({result.optimal_set}) = {manual_value:.6f}")
            
            # Check submodularity of our custom function (will benefit from cache)
            print(f"  Testing submodularity (with caching)...")
            check = solver.check_submodular(f6)
            if check.success:
                print(f"  Is submodular: {check.is_submodular}")
                if not check.is_submodular:
                    print(f"  Violations: {check.violations}")
                
                # Show updated cache stats
                final_stats = solver.get_cache_stats()
                additional_calls = final_stats['total_calls'] - cache_stats['total_calls']
                print(f"  Submodularity check added {additional_calls} more calls")
                print(f"  Final hit rate: {final_stats['hit_rate']:.1%}")
        else:
            print(f"  ❌ Error: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nTo run this demo:")
        print("1. Build the library: julia build_python_library.jl")
        print("2. Set library path environment variable")
        print("3. Run: python3 submodular_minimization_python.py")


if __name__ == "__main__":
    demo()