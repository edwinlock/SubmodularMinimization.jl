"""
Built-in submodular function implementations.

This module provides Python wrappers for the Julia implementations of common 
submodular functions, eliminating code duplication while providing a Pythonic interface.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import numpy.typing as npt

from .core import SubmodularFunction, BoolArray
from .julia_bridge import get_julia_bridge
from .exceptions import PySubmodularError


class ConcaveFunction(SubmodularFunction):
    """
    Concave submodular function: f(S) = |S|^α where 0 < α < 1.
    
    This is a classic example of a submodular function commonly used
    in testing and theoretical analysis.
    
    Args:
        n: Size of the ground set
        alpha: Concavity parameter (must be in (0, 1))
        
    Example:
        >>> f = ConcaveFunction(n=10, alpha=0.7)
        >>> subset = np.array([True, False, True, False, False, True, False, False, False, False])
        >>> value = f.evaluate(subset)  # Returns 3^0.7 ≈ 2.28
    """
    
    def __init__(self, n: int, alpha: float) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"Alpha must be in (0, 1) for submodularity, got {alpha}")
        
        super().__init__(n)
        self.alpha = alpha
        
        # Create corresponding Julia function
        bridge = get_julia_bridge()
        self._julia_func = bridge.call_julia_function(
            "ConcaveSubmodularFunction", n, alpha
        )
    
    def evaluate(self, subset: BoolArray) -> float:
        """Evaluate f(S) = |S|^α using Julia implementation."""
        self._validate_subset(subset)
        
        # Convert boolean array to Julia BitVector (1-indexed)
        bridge = get_julia_bridge()
        julia_bitvector = bridge.convert_python_array(subset)
        
        # Call Julia evaluate function
        result = bridge.call_julia_function("evaluate", self._julia_func, julia_bitvector)
        return float(result)


class SquareRootFunction(SubmodularFunction):
    """
    Square root function: f(S) = √|S|.
    
    This is equivalent to ConcaveFunction with α = 0.5.
    
    Args:
        n: Size of the ground set
        
    Example:
        >>> f = SquareRootFunction(n=8)
        >>> subset = np.array([True, True, False, False, True, False, False, False])
        >>> value = f.evaluate(subset)  # Returns √3 ≈ 1.73
    """
    
    def __init__(self, n: int) -> None:
        super().__init__(n)
        
        # Create corresponding Julia function
        bridge = get_julia_bridge()
        self._julia_func = bridge.call_julia_function("SquareRootFunction", n)
    
    def evaluate(self, subset: BoolArray) -> float:
        """Evaluate f(S) = √|S| using Julia implementation."""
        self._validate_subset(subset)
        
        # Convert boolean array to Julia BitVector
        bridge = get_julia_bridge()
        julia_bitvector = bridge.convert_python_array(subset)
        
        # Call Julia evaluate function
        result = bridge.call_julia_function("evaluate", self._julia_func, julia_bitvector)
        return float(result)


class MatroidRankFunction(SubmodularFunction):
    """
    Matroid rank function: f(S) = min(|S|, k).
    
    This represents the rank function of a uniform matroid, where
    at most k elements can be selected.
    
    Args:
        n: Size of the ground set
        k: Rank constraint (maximum number of elements)
        
    Example:
        >>> f = MatroidRankFunction(n=10, k=3)
        >>> subset = np.array([True, True, True, True, True, False, False, False, False, False])
        >>> value = f.evaluate(subset)  # Returns min(5, 3) = 3
    """
    
    def __init__(self, n: int, k: int) -> None:
        if not (0 <= k <= n):
            raise ValueError(f"Rank k must be in [0, {n}], got {k}")
        
        super().__init__(n)
        self.k = k
        
        # Create corresponding Julia function
        bridge = get_julia_bridge()
        self._julia_func = bridge.call_julia_function("MatroidRankFunction", n, k)
    
    def evaluate(self, subset: BoolArray) -> float:
        """Evaluate f(S) = min(|S|, k) using Julia implementation."""
        self._validate_subset(subset)
        
        # Convert boolean array to Julia BitVector
        bridge = get_julia_bridge()
        julia_bitvector = bridge.convert_python_array(subset)
        
        # Call Julia evaluate function
        result = bridge.call_julia_function("evaluate", self._julia_func, julia_bitvector)
        return float(result)


class CutFunction(SubmodularFunction):
    """
    Graph cut function: f(S) = number of edges between S and complement of S.
    
    This function counts the number of edges in the cut induced by the subset S.
    
    Args:
        n: Number of vertices in the graph
        edges: List of edges as (u, v) tuples (0-indexed)
        
    Example:
        >>> edges = [(0, 1), (1, 2), (2, 3), (0, 3)]  # Square graph
        >>> f = CutFunction(n=4, edges=edges)
        >>> subset = np.array([True, True, False, False])  # Split into {0,1} and {2,3}
        >>> value = f.evaluate(subset)  # Returns 2 (edges (1,2) and (0,3))
    """
    
    def __init__(self, n: int, edges: List[Tuple[int, int]]) -> None:
        super().__init__(n)
        
        # Validate edges
        for i, (u, v) in enumerate(edges):
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"Edge {i}: vertices must be in [0, {n})")
            if u == v:
                raise ValueError(f"Edge {i}: self-loops not allowed")
            if u >= v:
                raise ValueError(f"Edge {i}: edges should be ordered as (u, v) with u < v")
        
        self.edges = edges
        
        # Convert to Julia format (1-indexed)
        julia_edges = [(u + 1, v + 1) for u, v in edges]
        
        # Create corresponding Julia function
        bridge = get_julia_bridge()
        julia_edges_array = bridge.convert_python_array(julia_edges)
        self._julia_func = bridge.call_julia_function("CutFunction", n, julia_edges_array)
    
    def evaluate(self, subset: BoolArray) -> float:
        """Evaluate cut function using Julia implementation."""
        self._validate_subset(subset)
        
        # Convert boolean array to Julia BitVector
        bridge = get_julia_bridge()
        julia_bitvector = bridge.convert_python_array(subset)
        
        # Call Julia evaluate function
        result = bridge.call_julia_function("evaluate", self._julia_func, julia_bitvector)
        return float(result)


class FacilityLocationFunction(SubmodularFunction):
    """
    Facility location function: f(S) = Σᵢ max_{j∈S} wᵢⱼ.
    
    This function models facility location where customers are served
    by their closest selected facility.
    
    Args:
        n: Number of facilities
        weights: Weight matrix (customers × facilities)
        
    Example:
        >>> weights = np.random.rand(5, 8)  # 5 customers, 8 facilities
        >>> f = FacilityLocationFunction(n=8, weights=weights)
        >>> subset = np.array([True, False, True, False, False, True, False, False])
        >>> value = f.evaluate(subset)  # Sum of max benefits for each customer
    """
    
    def __init__(self, n: int, weights: npt.NDArray[np.floating]) -> None:
        super().__init__(n)
        
        weights = np.asarray(weights, dtype=float)
        
        if weights.ndim != 2:
            raise ValueError("Weights must be a 2D array")
        
        if weights.shape[1] != n:
            raise ValueError(f"Weight matrix must have {n} columns (facilities)")
        
        if weights.shape[0] == 0:
            raise ValueError("Weight matrix must have at least one row (customer)")
        
        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative")
        
        self.weights = weights
        
        # Create corresponding Julia function
        bridge = get_julia_bridge()
        julia_weights = bridge.convert_python_array(weights)
        self._julia_func = bridge.call_julia_function(
            "FacilityLocationFunction", n, julia_weights
        )
    
    def evaluate(self, subset: BoolArray) -> float:
        """Evaluate facility location function using Julia implementation."""
        self._validate_subset(subset)
        
        # Convert boolean array to Julia BitVector
        bridge = get_julia_bridge()
        julia_bitvector = bridge.convert_python_array(subset)
        
        # Call Julia evaluate function
        result = bridge.call_julia_function("evaluate", self._julia_func, julia_bitvector)
        return float(result)


class WeightedCoverageFunction(SubmodularFunction):
    """
    Weighted coverage function: f(S) = total weight of elements covered by sets in S.
    
    This function models set cover problems where each element has a weight
    and each set covers some subset of elements.
    
    Args:
        n: Number of sets
        element_weights: Weight of each element
        coverage_matrix: Boolean matrix where coverage_matrix[element, set] 
                        indicates if the set covers the element
    """
    
    def __init__(
        self, 
        n: int, 
        element_weights: npt.NDArray[np.floating],
        coverage_matrix: npt.NDArray[np.bool_]
    ) -> None:
        super().__init__(n)
        
        element_weights = np.asarray(element_weights, dtype=float)
        coverage_matrix = np.asarray(coverage_matrix, dtype=bool)
        
        if element_weights.ndim != 1:
            raise ValueError("Element weights must be a 1D array")
        
        if coverage_matrix.ndim != 2:
            raise ValueError("Coverage matrix must be 2D")
        
        if coverage_matrix.shape[1] != n:
            raise ValueError(f"Coverage matrix must have {n} columns (sets)")
        
        if len(element_weights) != coverage_matrix.shape[0]:
            raise ValueError("Element weights length must match coverage matrix rows")
        
        if np.any(element_weights < 0):
            raise ValueError("All element weights must be non-negative")
        
        self.element_weights = element_weights
        self.coverage_matrix = coverage_matrix
        
        # Create corresponding Julia function
        bridge = get_julia_bridge()
        julia_weights = bridge.convert_python_array(element_weights)
        julia_coverage = bridge.convert_python_array(coverage_matrix)
        
        self._julia_func = bridge.call_julia_function(
            "WeightedCoverageFunction", n, julia_weights, julia_coverage
        )
    
    def evaluate(self, subset: BoolArray) -> float:
        """Evaluate weighted coverage function using Julia implementation."""
        self._validate_subset(subset)
        
        # Convert boolean array to Julia BitVector
        bridge = get_julia_bridge()
        julia_bitvector = bridge.convert_python_array(subset)
        
        # Call Julia evaluate function
        result = bridge.call_julia_function("evaluate", self._julia_func, julia_bitvector)
        return float(result)


class LogDeterminantFunction(SubmodularFunction):
    """
    Log-determinant function: f(S) = log det(A_S + εI) - log det(εI).
    
    This function is used in experimental design and sensor placement.
    The matrix A should be positive semidefinite.
    
    Args:
        n: Size of the ground set
        A: Positive semidefinite matrix (n × n)
        epsilon: Regularization parameter (default: 1e-6)
    """
    
    def __init__(
        self, 
        n: int, 
        A: npt.NDArray[np.floating], 
        epsilon: float = 1e-6
    ) -> None:
        super().__init__(n)
        
        A = np.asarray(A, dtype=float)
        
        if A.shape != (n, n):
            raise ValueError(f"Matrix A must be {n}×{n}")
        
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # Check if A is approximately symmetric
        if not np.allclose(A, A.T, rtol=1e-10):
            raise ValueError("Matrix A must be symmetric")
        
        # Check if A is positive semidefinite (approximately)
        eigenvals = np.linalg.eigvals(A)
        if np.any(eigenvals < -1e-10):
            raise ValueError("Matrix A must be positive semidefinite")
        
        self.A = A
        self.epsilon = epsilon
        
        # Create corresponding Julia function
        bridge = get_julia_bridge()
        julia_A = bridge.convert_python_array(A)
        self._julia_func = bridge.call_julia_function(
            "LogDeterminantFunction", n, julia_A, epsilon
        )
    
    def evaluate(self, subset: BoolArray) -> float:
        """Evaluate log-determinant function using Julia implementation."""
        self._validate_subset(subset)
        
        # Convert boolean array to Julia BitVector
        bridge = get_julia_bridge()
        julia_bitvector = bridge.convert_python_array(subset)
        
        # Call Julia evaluate function
        result = bridge.call_julia_function("evaluate", self._julia_func, julia_bitvector)
        return float(result)


class EntropyFunction(SubmodularFunction):
    """
    Entropy-based submodular function.
    
    This function models information-theoretic quantities and is commonly
    used in feature selection and active learning.
    
    Args:
        n: Number of variables
        probabilities: Probability matrix (outcomes × variables)
    """
    
    def __init__(self, n: int, probabilities: npt.NDArray[np.floating]) -> None:
        super().__init__(n)
        
        probabilities = np.asarray(probabilities, dtype=float)
        
        if probabilities.ndim != 2:
            raise ValueError("Probabilities must be a 2D array")
        
        if probabilities.shape[1] != n:
            raise ValueError(f"Probability matrix must have {n} columns (variables)")
        
        if probabilities.shape[0] == 0:
            raise ValueError("Probability matrix must have at least one row (outcome)")
        
        # Validate probabilities
        for j in range(n):
            col_sum = np.sum(probabilities[:, j])
            if not np.isclose(col_sum, 1.0, atol=1e-10):
                raise ValueError(f"Column {j} probabilities must sum to 1, got {col_sum}")
            
            if np.any(probabilities[:, j] < 0):
                raise ValueError("All probabilities must be non-negative")
        
        self.probabilities = probabilities
        
        # Create corresponding Julia function
        bridge = get_julia_bridge()
        julia_probs = bridge.convert_python_array(probabilities)
        self._julia_func = bridge.call_julia_function(
            "EntropyFunction", n, julia_probs
        )
    
    def evaluate(self, subset: BoolArray) -> float:
        """Evaluate entropy function using Julia implementation."""
        self._validate_subset(subset)
        
        # Convert boolean array to Julia BitVector
        bridge = get_julia_bridge()
        julia_bitvector = bridge.convert_python_array(subset)
        
        # Call Julia evaluate function
        result = bridge.call_julia_function("evaluate", self._julia_func, julia_bitvector)
        return float(result)