# PySubmodular

**Python wrapper for SubmodularMinimization.jl**

PySubmodular provides a high-performance Python interface to the Julia [SubmodularMinimization.jl](https://github.com/edwlock/SubmodularMinimization.jl) package, enabling efficient submodular function minimization and general convex optimization using Wolfe's algorithm.

## Features

- 🚀 **High Performance**: Leverages Julia's optimized implementation
- 🐍 **Pythonic API**: Idiomatic Python interface with type hints
- 📊 **Rich Function Library**: Built-in implementations of common submodular functions
- 🔧 **Flexible**: Direct access to Wolfe's algorithm for custom polytopes
- 🧪 **Well Tested**: Comprehensive test suite with correctness verification
- 📝 **Fully Documented**: Complete API documentation and examples

## Installation

### Prerequisites

1. **Julia**: Install Julia ≥ 1.6 from [julialang.org](https://julialang.org/downloads/)
2. **PyJulia**: Install the Python-Julia interface

```bash
pip install julia
```

3. **Configure PyJulia** (one-time setup):

```python
import julia
julia.install()
```

### Install PySubmodular

```bash
pip install pysubmodular
```

For development installation:

```bash
git clone https://github.com/edwlock/SubmodularMinimization.jl
cd SubmodularMinimization.jl/pysubmodular
pip install -e .
```

## Quick Start

### Basic Submodular Minimization

```python
import numpy as np
import pysubmodular as psm

# Create a concave submodular function f(S) = |S|^0.7
f = psm.ConcaveFunction(n=10, alpha=0.7)

# Minimize the function
result = psm.minimize_submodular(f)

print(f"Minimum value: {result.min_value:.6f}")
print(f"Minimizer set: {result.selected_indices}")
print(f"Converged in {result.iterations} iterations")
```

### Using Wolfe's Algorithm Directly

```python
import numpy as np
import pysubmodular as psm

# Define a linear oracle for the standard simplex
def simplex_oracle(c):
    \"\"\"Oracle for {x: x ≥ 0, sum(x) = 1}\"\"\"
    i_min = np.argmin(c)
    vertex = np.zeros(len(c))
    vertex[i_min] = 1.0
    return vertex

# Find minimum norm point in the simplex
result = psm.wolfe_algorithm(simplex_oracle, dimension=5)
print(f"Minimum norm point: {result.min_norm_point}")
print(f"Norm: {result.norm_value:.6f}")
```

## Built-in Functions

PySubmodular includes Python implementations of common submodular functions:

### Mathematical Functions

```python
# Concave functions: f(S) = |S|^α
f1 = psm.ConcaveFunction(n=10, alpha=0.7)

# Square root: f(S) = √|S|
f2 = psm.SquareRootFunction(n=8)

# Matroid rank: f(S) = min(|S|, k)
f3 = psm.MatroidRankFunction(n=10, k=5)
```

### Graph Functions

```python
# Cut function
edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
f4 = psm.CutFunction(n=4, edges=edges)
```

### Machine Learning Functions

```python
# Facility location
weights = np.random.rand(5, 8)  # 5 customers, 8 facilities
f5 = psm.FacilityLocationFunction(n=8, weights=weights)

# Weighted coverage
element_weights = np.random.rand(10)
coverage_matrix = np.random.rand(10, 6) < 0.3  # 6 sets, 10 elements
f6 = psm.WeightedCoverageFunction(n=6, element_weights=element_weights, 
                                 coverage_matrix=coverage_matrix)
```

### Information Theory Functions

```python
# Log-determinant (experimental design)
A = np.random.rand(5, 5)
A = A @ A.T  # Make positive semidefinite
f7 = psm.LogDeterminantFunction(n=5, A=A)

# Entropy function
probabilities = np.random.rand(4, 6)
probabilities = probabilities / probabilities.sum(axis=0)  # Normalize
f8 = psm.EntropyFunction(n=6, probabilities=probabilities)
```

## Advanced Usage

### Custom Submodular Functions

```python
class MySubmodularFunction(psm.SubmodularFunction):
    def __init__(self, n, parameter):
        super().__init__(n)
        self.parameter = parameter
    
    def evaluate(self, subset):
        # Implement your submodular function
        # Must satisfy f(S) + f(T) ≥ f(S∪T) + f(S∩T)
        return my_computation(subset, self.parameter)

# Use with the optimization algorithms
f = MySubmodularFunction(n=8, parameter=1.5)
result = psm.minimize_submodular(f)
```

### Performance Optimization

```python
# Use pre-allocation for better performance
result = psm.minimize_submodular(f, use_preallocation=True)

# Custom tolerance and iteration limits
result = psm.minimize_submodular(f, tolerance=1e-8, max_iterations=5000)
```

### Error Handling

```python
from pysubmodular.exceptions import OptimizationError, PySubmodularError

try:
    result = psm.minimize_submodular(f)
except OptimizationError as e:
    print(f"Optimization failed: {e}")
except PySubmodularError as e:
    print(f"PySubmodular error: {e}")
```

## API Reference

### Core Functions

- `minimize_submodular(func, **kwargs)`: Minimize a submodular function
- `wolfe_algorithm(oracle, dimension, **kwargs)`: Find minimum norm point in polytope

### Result Classes

- `OptimizationResult`: Result of submodular minimization
- `WolfeResult`: Result of Wolfe's algorithm

### Built-in Functions

- `ConcaveFunction`: f(S) = |S|^α
- `SquareRootFunction`: f(S) = √|S|
- `MatroidRankFunction`: f(S) = min(|S|, k)
- `CutFunction`: Graph cut function
- `FacilityLocationFunction`: Facility location objective
- `WeightedCoverageFunction`: Weighted set cover
- `LogDeterminantFunction`: Log-determinant for experimental design
- `EntropyFunction`: Information-theoretic function

### Base Classes

- `SubmodularFunction`: Abstract base class for submodular functions

## Examples

See the `examples/` directory for complete examples:

- `basic_optimization.py`: Basic submodular minimization
- `custom_functions.py`: Implementing custom submodular functions
- `wolfe_algorithm.py`: Using Wolfe's algorithm directly
- `machine_learning.py`: ML applications (facility location, feature selection)
- `graph_problems.py`: Graph-based optimization problems

## Performance

PySubmodular achieves high performance by:

- **Julia Backend**: Leverages Julia's optimized numerical computing
- **Pre-allocation**: Memory-efficient workspace reuse
- **Vectorized Operations**: NumPy integration for array operations
- **Minimal Overhead**: Efficient Python-Julia communication

Typical performance (n=20, tolerance=1e-6):
- **Concave functions**: ~1ms
- **Cut functions**: ~2ms  
- **Facility location**: ~5ms
- **Complex functions**: ~10ms

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pysubmodular

# Run specific test categories
pytest tests/test_functions.py
pytest tests/test_core.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

GNU General Public License v3.0 - see [LICENSE](../LICENSE) file for details.

## References

1. **Chakrabarty, Jain, Kothari (2014)**: "Provable Submodular Minimization using Wolfe's Algorithm"
2. **Fujishige (1980)**: "Lexicographically optimal base of a polyhedron"
3. **Wolfe (1976)**: "Finding the nearest point in a polytope"

## Citation

If you use PySubmodular in your research, please cite:

```bibtex
@software{pysubmodular2024,
  title={PySubmodular: Python Interface for High-Performance Submodular Minimization},
  author={Wlock, Edwin},
  year={2024},
  url={https://github.com/edwlock/SubmodularMinimization.jl}
}
```