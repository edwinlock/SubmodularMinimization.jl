"""
Tests for built-in submodular functions.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import pysubmodular as psm
from pysubmodular.exceptions import PySubmodularError, DimensionError


class TestConcaveFunction:
    """Test ConcaveFunction class."""
    
    def test_initialization(self):
        """Test function initialization."""
        f = psm.ConcaveFunction(n=5, alpha=0.7)
        assert f.n == 5
        assert f.alpha == 0.7
    
    def test_invalid_alpha(self):
        """Test invalid alpha values."""
        with pytest.raises(ValueError, match="Alpha must be in \\(0, 1\\)"):
            psm.ConcaveFunction(n=5, alpha=0.0)
        
        with pytest.raises(ValueError, match="Alpha must be in \\(0, 1\\)"):
            psm.ConcaveFunction(n=5, alpha=1.0)
        
        with pytest.raises(ValueError, match="Alpha must be in \\(0, 1\\)"):
            psm.ConcaveFunction(n=5, alpha=1.5)
    
    def test_evaluation(self):
        """Test function evaluation."""
        f = psm.ConcaveFunction(n=4, alpha=0.5)
        
        # Empty set
        subset = np.array([False, False, False, False])
        assert f.evaluate(subset) == 0.0
        
        # Single element
        subset = np.array([True, False, False, False])
        assert_allclose(f.evaluate(subset), 1.0)
        
        # Multiple elements
        subset = np.array([True, True, True, False])
        assert_allclose(f.evaluate(subset), np.sqrt(3))
        
        # Full set
        subset = np.array([True, True, True, True])
        assert_allclose(f.evaluate(subset), 2.0)
    
    def test_submodularity(self):
        """Test that the function is submodular."""
        f = psm.ConcaveFunction(n=6, alpha=0.8)
        assert f.check_submodularity(num_samples=50)
    
    def test_marginal_values(self):
        """Test marginal value computation."""
        f = psm.ConcaveFunction(n=4, alpha=0.5)
        
        subset = np.array([True, False, True, False])  # {0, 2}
        
        # Adding element 1
        marginal = f.marginal_value(subset, 1)
        expected = np.sqrt(3) - np.sqrt(2)  # 3^0.5 - 2^0.5
        assert_allclose(marginal, expected)
        
        # Adding element that's already there
        marginal = f.marginal_value(subset, 0)
        assert_allclose(marginal, 0.0)


class TestSquareRootFunction:
    """Test SquareRootFunction class."""
    
    def test_initialization(self):
        """Test function initialization."""
        f = psm.SquareRootFunction(n=8)
        assert f.n == 8
    
    def test_evaluation(self):
        """Test function evaluation."""
        f = psm.SquareRootFunction(n=5)
        
        # Test various subset sizes
        for k in range(6):
            subset = np.zeros(5, dtype=bool)
            subset[:k] = True
            
            expected = np.sqrt(k)
            assert_allclose(f.evaluate(subset), expected)
    
    def test_equivalence_to_concave(self):
        """Test equivalence to ConcaveFunction with alpha=0.5."""
        n = 6
        f_sqrt = psm.SquareRootFunction(n)
        f_concave = psm.ConcaveFunction(n, alpha=0.5)
        
        # Test on random subsets
        np.random.seed(42)
        for _ in range(10):
            subset = np.random.rand(n) < 0.5
            assert_allclose(f_sqrt.evaluate(subset), f_concave.evaluate(subset))


class TestMatroidRankFunction:
    """Test MatroidRankFunction class."""
    
    def test_initialization(self):
        """Test function initialization."""
        f = psm.MatroidRankFunction(n=10, k=5)
        assert f.n == 10
        assert f.k == 5
    
    def test_invalid_k(self):
        """Test invalid k values."""
        with pytest.raises(ValueError, match="Rank k must be in"):
            psm.MatroidRankFunction(n=5, k=-1)
        
        with pytest.raises(ValueError, match="Rank k must be in"):
            psm.MatroidRankFunction(n=5, k=6)
    
    def test_evaluation(self):
        """Test function evaluation."""
        f = psm.MatroidRankFunction(n=8, k=3)
        
        # Small subsets (less than k)
        subset = np.array([True, True, False, False, False, False, False, False])
        assert f.evaluate(subset) == 2.0
        
        # Subset equal to k
        subset = np.array([True, True, True, False, False, False, False, False])
        assert f.evaluate(subset) == 3.0
        
        # Large subset (more than k)
        subset = np.array([True, True, True, True, True, False, False, False])
        assert f.evaluate(subset) == 3.0  # Capped at k
        
        # Full set
        subset = np.ones(8, dtype=bool)
        assert f.evaluate(subset) == 3.0  # Capped at k
    
    def test_submodularity(self):
        """Test that the function is submodular."""
        f = psm.MatroidRankFunction(n=8, k=4)
        assert f.check_submodularity(num_samples=50)


class TestCutFunction:
    """Test CutFunction class."""
    
    def test_initialization(self):
        """Test function initialization."""
        edges = [(0, 1), (1, 2), (2, 3)]
        f = psm.CutFunction(n=4, edges=edges)
        assert f.n == 4
        assert f.edges == edges
    
    def test_invalid_edges(self):
        """Test invalid edge specifications."""
        # Out of bounds
        with pytest.raises(ValueError, match="vertices must be in"):
            psm.CutFunction(n=3, edges=[(0, 3)])
        
        # Self-loop
        with pytest.raises(ValueError, match="self-loops not allowed"):
            psm.CutFunction(n=3, edges=[(1, 1)])
        
        # Wrong order
        with pytest.raises(ValueError, match="edges should be ordered"):
            psm.CutFunction(n=3, edges=[(2, 1)])
    
    def test_evaluation_path_graph(self):
        """Test evaluation on a path graph."""
        # Path: 0-1-2-3
        edges = [(0, 1), (1, 2), (2, 3)]
        f = psm.CutFunction(n=4, edges=edges)
        
        # Empty cut
        subset = np.array([False, False, False, False])
        assert f.evaluate(subset) == 0.0
        
        # Single vertex
        subset = np.array([True, False, False, False])
        assert f.evaluate(subset) == 1.0  # Edge (0,1)
        
        # Split in middle
        subset = np.array([True, True, False, False])
        assert f.evaluate(subset) == 1.0  # Edge (1,2)
        
        # Full set
        subset = np.array([True, True, True, True])
        assert f.evaluate(subset) == 0.0  # No cut edges
    
    def test_evaluation_complete_graph(self):
        """Test evaluation on a complete graph."""
        # Complete graph on 4 vertices
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        f = psm.CutFunction(n=4, edges=edges)
        
        # Single vertex vs rest
        subset = np.array([True, False, False, False])
        assert f.evaluate(subset) == 3.0  # 3 edges from vertex 0
        
        # Two vs two
        subset = np.array([True, True, False, False])
        assert f.evaluate(subset) == 4.0  # 4 edges between {0,1} and {2,3}
    
    def test_submodularity(self):
        """Test that the function is submodular."""
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        f = psm.CutFunction(n=4, edges=edges)
        assert f.check_submodularity(num_samples=50)


class TestFacilityLocationFunction:
    """Test FacilityLocationFunction class."""
    
    def test_initialization(self):
        """Test function initialization."""
        weights = np.array([[1.0, 2.0, 3.0],
                           [2.0, 1.0, 2.0]])  # 2 customers, 3 facilities
        f = psm.FacilityLocationFunction(n=3, weights=weights)
        assert f.n == 3
        assert_array_equal(f.weights, weights)
    
    def test_invalid_weights(self):
        """Test invalid weight matrices."""
        # Wrong number of columns
        with pytest.raises(ValueError, match="must have 3 columns"):
            weights = np.array([[1.0, 2.0]])
            psm.FacilityLocationFunction(n=3, weights=weights)
        
        # Negative weights
        with pytest.raises(ValueError, match="must be non-negative"):
            weights = np.array([[1.0, -2.0, 3.0]])
            psm.FacilityLocationFunction(n=3, weights=weights)
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="must be a 2D array"):
            weights = np.array([1.0, 2.0, 3.0])
            psm.FacilityLocationFunction(n=3, weights=weights)
    
    def test_evaluation(self):
        """Test function evaluation."""
        weights = np.array([[1.0, 3.0, 2.0],
                           [2.0, 1.0, 4.0]])  # 2 customers, 3 facilities
        f = psm.FacilityLocationFunction(n=3, weights=weights)
        
        # Empty set
        subset = np.array([False, False, False])
        assert f.evaluate(subset) == 0.0
        
        # Single facility
        subset = np.array([True, False, False])  # Facility 0
        expected = 1.0 + 2.0  # Best for each customer
        assert_allclose(f.evaluate(subset), expected)
        
        # Multiple facilities
        subset = np.array([True, True, False])  # Facilities 0,1
        expected = max(1.0, 3.0) + max(2.0, 1.0)  # 3.0 + 2.0 = 5.0
        assert_allclose(f.evaluate(subset), expected)
        
        # All facilities
        subset = np.array([True, True, True])
        expected = max(1.0, 3.0, 2.0) + max(2.0, 1.0, 4.0)  # 3.0 + 4.0 = 7.0
        assert_allclose(f.evaluate(subset), expected)
    
    def test_submodularity(self):
        """Test that the function is submodular."""
        np.random.seed(42)
        weights = np.random.rand(4, 6)  # 4 customers, 6 facilities
        f = psm.FacilityLocationFunction(n=6, weights=weights)
        assert f.check_submodularity(num_samples=50)


class TestSubmodularFunctionBase:
    """Test SubmodularFunction base class functionality."""
    
    def test_invalid_ground_set_size(self):
        """Test invalid ground set sizes."""
        with pytest.raises(ValueError, match="must be positive"):
            psm.ConcaveFunction(n=0, alpha=0.5)
        
        with pytest.raises(ValueError, match="must be positive"):
            psm.ConcaveFunction(n=-1, alpha=0.5)
    
    def test_dimension_validation(self):
        """Test subset dimension validation."""
        f = psm.ConcaveFunction(n=5, alpha=0.7)
        
        # Wrong size
        with pytest.raises(DimensionError, match="Expected dimension 5, got 3"):
            f.evaluate(np.array([True, False, True]))
        
        # Wrong type
        with pytest.raises(TypeError, match="must be a boolean array"):
            f.evaluate(np.array([1, 0, 1, 0, 1]))
    
    def test_callable_interface(self):
        """Test that functions are callable."""
        f = psm.ConcaveFunction(n=3, alpha=0.5)
        subset = np.array([True, False, True])
        
        # Both should give same result
        result1 = f.evaluate(subset)
        result2 = f(subset)
        assert_allclose(result1, result2)
    
    def test_marginal_value_edge_cases(self):
        """Test marginal value computation edge cases."""
        f = psm.ConcaveFunction(n=4, alpha=0.5)
        subset = np.array([True, False, True, False])
        
        # Invalid element index
        with pytest.raises(ValueError, match="Element index must be"):
            f.marginal_value(subset, -1)
        
        with pytest.raises(ValueError, match="Element index must be"):
            f.marginal_value(subset, 4)
        
        # Element already in subset
        marginal = f.marginal_value(subset, 0)
        assert_allclose(marginal, 0.0)


if __name__ == "__main__":
    pytest.main([__file__])