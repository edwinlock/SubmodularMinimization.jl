# SubmodularMinimization.jl

A high-performance Julia package for submodular function minimization using the Fujishige-Wolfe algorithm with non-trivial optimizations and efforts to guarantee numerical stability.


## Overview

This package provides an implementation of the Fujishige-Wolfe algorithm for minimizing submodular functions. The implementation follows "Provable Submodular Minimization using Wolfe's Algorithm" by Chakrabarty, Jain, and Kothari (2014) with significant performance optimizations including matrix storage, numerical stability enhancements, and proper tolerance management.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/edwock/SubmodularMinimization.jl")
```

## Quick Start

```julia
using SubmodularMinimization

# Create a test function (see examples.jl for different example submodular functions)
f = ConcaveSubmodularFunction(10, 0.7)

# Main algorithm with automatic workspace management
S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f)
println("Minimum value: $min_val, Minimizing set: $(findall(S_min))")

# Using different tolerance for higher precision
S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=TIGHT_TOLERANCE)

# Pre-allocated version for maximum performance
workspace = WolfeWorkspace(10)
S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization!(workspace, f)
```

## Tolerances

The package provides standardized tolerance constants for consistent numerical behavior:

```julia
# Use predefined constants for consistent behavior
result = fujishige_wolfe_submodular_minimization(f; ε=DEFAULT_TOLERANCE)  # 1e-6 (balanced)
result = fujishige_wolfe_submodular_minimization(f; ε=LOOSE_TOLERANCE)    # 1e-4 (fast)  
result = fujishige_wolfe_submodular_minimization(f; ε=TIGHT_TOLERANCE)    # 5e-7 (precise)
```

## Performance Characteristics

### Benchmarking

```julia
# Run comprehensive performance analysis
julia performance_analysis.jl

# Quick performance test
julia -e "include(\"performance_analysis.jl\"); main()"
```

## Example Functions

The package includes many common submodular functions, all normalized so that f(∅) = 0:

```julia
# Basic functions
f1 = ConcaveSubmodularFunction(10, 0.7)  # f(S) = |S|^α
f2 = SquareRootFunction(8)               # f(S) = √|S|
f3 = MatroidRankFunction(10, 5)          # f(S) = min(|S|, k)

# Graph-based functions
edges = [(1,2), (2,3), (3,4), (1,4)]
f4 = CutFunction(4, edges)               # Graph cut function
f5 = create_random_cut_function(15, 0.3) # Random graph cuts

# Machine learning functions
weights = rand(5, 8)  # 5 customers, 8 facilities
f6 = FacilityLocationFunction(8, weights)           # Facility location
f7 = create_random_facility_location(10, 4)        # Random version

# Information theory
f8 = create_random_entropy_function(6, 4)          # Entropy-based
f9 = create_wishart_log_determinant(5)             # Log-determinant

# Combinatorial optimization  
f10 = create_random_coverage_function(8, 12)       # Weighted coverage
```

## Advanced Usage

### Performance-Critical Applications

```julia
# Pre-allocate workspace for repeated use (batch processing)
problems = [ConcaveSubmodularFunction(20, 0.5 + 0.3*rand()) for _ in 1:16]
workspace = WolfeWorkspace(20)  # Reuse for all problems with same n

# Process batch efficiently - no allocation overhead
results = [fujishige_wolfe_submodular_minimization!(workspace, p) for p in problems]

# Or with explicit loop for custom processing
for problem in problems
    S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization!(workspace, problem)
    # Process results...
end
```

## Package Structure

```
SubmodularMinimization.jl/
├── src/
│   ├── SubmodularMinimization.jl  # Main module with tolerance constants
│   ├── core.jl                           # Abstract types and interfaces
│   ├── examples.jl                       # Example submodular functions
│   ├── oracles.jl                        # Linear optimization and affine minimization
│   ├── algorithms.jl                     # Fujishige-Wolfe implementation
│   └── utils.jl                          # Utilities and testing functions
├── test/                                  # Comprehensive test suite (885 tests)
└── test_verbose.jl                       # Verbose test runner
```

## Testing

```bash
# Run full test suite (885 tests)
julia test/runtests.jl

# Verbose testing with progress output
julia test_verbose.jl

# Run specific test categories
julia -e "using Pkg; Pkg.activate(\".\"); using Test; using SubmodularMinimization; include(\"test/test_algorithms.jl\")"
```

## Dependencies

- **LinearAlgebra**: Matrix operations and norms
- **Random**: Random number generation for testing and examples

## Citation

If you use this package in your research, please refer to this GitHub repository and cite:

```
@article{chakrabarty2014provable,
  title={Provable Submodular Minimization using Wolfe's Algorithm},
  author={Chakrabarty, Deeparnab and Jain, Prateek and Kothari, Pravesh},
  year={2014}
}
```

## Contributing

Contributions are welcome! Please see our contributing guidelines and ensure all tests pass:

```bash
julia test/runtests.jl      # All 885 tests must pass
julia performance_analysis.jl  # Performance regression check
```

## License

MIT License - see LICENSE file for details.

---

For detailed algorithmic documentation, implementation details, and theoretical background, see [README.txt](README.txt).