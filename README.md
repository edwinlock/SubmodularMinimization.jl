# SubmodularMinimization.jl

A high-performance Julia package for submodular function minimization using the Fujishige-Wolfe algorithm with non-trivial optimizations and efforts to guarantee numerical stability. 

The implementation follows "Provable Submodular Minimization using Wolfe's Algorithm" by Chakrabarty, Jain, and Kothari (2014).

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

## Algorithm Variants

The package provides multiple interfaces for different use cases:

| Function | Description | Use Case |
|----------|-------------|----------|
| `fujishige_wolfe_submodular_minimization()` | Complete submodular minimization with automatic workspace | General use, convenient interface |
| `fujishige_wolfe_submodular_minimization!()` | Pre-allocated workspace version | Performance-critical, minimal allocations |  
| `wolfe_algorithm()` / `wolfe_algorithm!()` | Wolfe algorithm for any polytope | Custom oracles, general convex optimization |
| `brute_force_minimization()` | Exhaustive search (2^n evaluations) | Testing, verification (n ≤ 20) |
| `is_submodular()` | Check if a function is submodular using Definition 3 | Validation, debugging (n ≤ 10) |
| `is_minimiser()` | Check if a given set is optimal for a submodular function | Solution verification (any n) |

## Tolerances

The package provides standardized tolerance constants for consistent numerical behavior:

```julia
# Use predefined constants for consistent behavior, or set your own ε.
result = fujishige_wolfe_submodular_minimization(f; ε=DEFAULT_TOLERANCE)  # 1e-6 (balanced)
result = fujishige_wolfe_submodular_minimization(f; ε=LOOSE_TOLERANCE)    # 1e-4 (fast)  
result = fujishige_wolfe_submodular_minimization(f; ε=TIGHT_TOLERANCE)    # 5e-7 (precise)
```

## Performance Characteristics

The implementation achieves excellent performance, handling problems with n=100 in milliseconds and scaling to n=150 with sub-second runtimes. Most practical problems solve in under 10ms with reliable convergence.

| Problem Type | Size | Time | Memory | Iterations | Notes |
|--------------|------|------|--------|------------|-------|
| Cut Functions (sparse) | n ≤ 30 | <0.01ms | ~61MB | 2 | Optimal in 2 iterations |
| Cut Functions (dense) | n = 50-120 | 3-10ms | ~66-72MB | 105-144 | Scales with edge density |
| Concave Functions | n = 5-25 | 0.01-0.1ms | ~61-62MB | 5-30 | Depends on α parameter |
| Concave Functions | n = 40-100 | 0.4-7ms | ~63-72MB | 54-129 | Linear scaling |
| Concave Functions | n = 150 | 34ms | ~101MB | 220 | Largest test case |
| Bipartite Matching | 3×3 to 15×15 | 0.005-1ms | ~61-63MB | 5-107 | Quadratic node pairs |
| Facility Location | 4+6 to 25+40 | 0.05-18ms | ~61-82MB | 26-190 | Scales with facilities×clients |
| Weighted Coverage | 6×8 to 50×60 | 0.02-10ms | ~61-70MB | 16-141 | Problem-dependent |
| Special Cases | n = 1-16 | <0.03ms | ~15-61MB | 1-16 | Trivial, sqrt, graphs |

### Benchmarking

```julia
# Run comprehensive performance analysis
julia performance_analysis.jl

# Quick performance test
julia -e "include(\"performance_analysis.jl\"); main()"
```

## Testing and Validation

### Submodularity Verification

Before applying the Fujishige-Wolfe algorithm, you should verify that your function is actually submodular. **Non-submodular functions will not be minimized correctly**, leading to suboptimal results.

```julia
# Check if a function is submodular
f = ConcaveSubmodularFunction(4, 0.5)
is_sub, violations, total_tests = is_submodular(f; verbose=true)
println("Is submodular: $is_sub")  # Output: true

# Feature Selection functions are typically NOT submodular
f_feat = create_feature_selection(4)
is_sub, violations, total_tests = is_submodular(f_feat; verbose=true) 
println("Is submodular: $is_sub")  # Output: false (all tests failed)

# Check without verbose output
is_sub, violations, total_tests = is_submodular(f)
if !is_sub
    @warn "Function is not submodular! Found $violations violations out of $total_tests tests."
end
```

**Important**: The `is_submodular()` function has O(n² × 2ⁿ) complexity and is only practical for small problems (n ≤ 10). For larger functions, submodularity must be verified theoretically or through sampling.

### Solution Verification with `is_minimiser()`

The `is_minimiser()` function provides an efficient O(n) method to verify whether a given solution is optimal for any submodular function, regardless of problem size:

```julia
f = ConcaveSubmodularFunction(20, 0.7)

# Get algorithm result
S_alg, val_alg = fujishige_wolfe_submodular_minimization(f)

# Verify optimality (works for any n)
is_optimal, improvement, better_val = is_minimiser(S_alg, f; verbose=true)

if is_optimal
    println("✓ Solution is globally optimal!")
else
    println("✗ Solution is not optimal. $improvement")
    println("  Current value: $val_alg")
    println("  Better value: $better_val")
end
```

**Key advantages of `is_minimiser()`:**
- **Scalable**: O(n) complexity vs O(2ⁿ) for brute force
- **Works for any problem size**: No exponential growth like brute force
- **Provides improvement hints**: When not optimal, suggests specific improvements
- **Supports both input formats**: BitVector or Vector{Int} (indices of selected elements)

```julia
# Using Vector{Int} format (indices of selected elements)
selected_elements = [1, 3, 5]  # Elements in the set
is_optimal, improvement, better_val = is_minimiser(selected_elements, f, f.n)

# Using BitVector format
S = falses(f.n)
S[[1, 3, 5]] .= true
is_optimal, improvement, better_val = is_minimiser(S, f)
```

### Brute Force Verification

For small problems, you can verify the correctness of the algorithm using brute force search:

```julia
f = ConcaveSubmodularFunction(6, 0.7)

# Algorithm result
S_alg, val_alg = fujishige_wolfe_submodular_minimization(f)

# Brute force verification (for n ≤ 20)
S_bf, val_bf = brute_force_minimization(f)

# Compare results
println("Algorithm: value = $val_alg, set = $(findall(S_alg))")
println("Brute force: value = $val_bf, set = $(findall(S_bf))")
println("Match: $(abs(val_alg - val_bf) < COMPARISON_TOLERANCE)")

# Verify both solutions are optimal
println("Algorithm optimal: $(is_minimiser(S_alg, f)[1])")
println("Brute force optimal: $(is_minimiser(S_bf, f)[1])")
```

## Example Functions

The package includes a comprehensive collection of submodular functions covering major application areas, all normalized so that f(∅) = 0:

### Basic Mathematical Functions

```julia
# Concave functions: f(S) = |S|^α (0 < α < 1)
f1 = ConcaveSubmodularFunction(10, 0.7)

# Square root function: f(S) = √|S|
f2 = SquareRootFunction(8)

# Matroid rank function: f(S) = min(|S|, k)
f3 = MatroidRankFunction(10, 5)
```

### Graph Theory Functions

```julia
# Cut functions (graph partitioning)
edges = [(1,2), (2,3), (3,4), (1,4)]
f4 = CutFunction(4, edges)
f5 = create_random_cut_function(15, 0.3)

# Asymmetric cut functions (directional penalties)
f6 = create_asymmetric_cut(8, 0.4, 2.5)

# Bipartite matching functions
f7 = create_bipartite_matching(5, 6, 0.4)
```

### Machine Learning & Optimization

```julia
# Facility location (service optimization)
f8 = create_random_facility_location(10, 8)
weights = rand(5, 8)  # Custom weights
f9 = FacilityLocationFunction(8, weights)

# Weighted coverage (set cover variants)
f10 = create_random_coverage_function(8, 12)

# Feature selection (mutual information-based)
f11 = create_feature_selection(10; correlation_strength=0.4, α=0.6)

# Diversity maximization (recommendation systems)
f12 = create_diversity_function(8; similarity_strength=0.3)
```

### Information Theory & Experimental Design

```julia
# Entropy functions (information selection)
f13 = create_random_entropy_function(6, 4)

# Log-determinant functions (optimal experimental design)
f14 = create_wishart_log_determinant(5)

# Information gain (active learning)
f15 = create_information_gain(8; correlation_strength=0.3, decay=0.4)

# Sensor placement (monitoring networks)
f16 = create_sensor_placement(10, 15; coverage_prob=0.3)
```

### Computer Vision & AI

```julia
# Image segmentation (graph-based)
f17 = create_image_segmentation(12; edge_density=0.3, λ=1.0)

# Feature selection for ML
f18 = create_feature_selection(8; correlation_strength=0.5, α=0.6)
```

### Economics & Auction Theory

```julia
# Gross substitutes (auction theory)
f19 = create_gross_substitutes(6; substitutability_strength=0.3)

# Auction revenue optimization
f20 = create_auction_revenue(8, 5; competition_strength=0.2)

# Market share functions (product portfolios)  
f21 = create_market_share(6, 8; cannibalization_strength=0.25)
```

### Advanced Specialized Functions

```julia
# Concave with penalty (avoiding trivial solutions)
f22 = create_concave_with_penalty(8, 0.7, 3.0)

# Custom functions - implement SubmodularFunction interface
struct MyFunction <: SubmodularFunction
    n::Int
    parameters::MyParams
end

ground_set_size(f::MyFunction) = f.n
evaluate(f::MyFunction, S::BitVector) = my_computation(S, f.parameters)
```

All functions support the same interface and can be used interchangeably with the optimization algorithms.

## Direct Use of Wolfe's Algorithm

Beyond submodular minimization, this package provides direct access to **Wolfe's algorithm** for finding the minimum norm point in any convex polytope defined by a linear optimization oracle. This is useful for general convex optimization problems.

### Basic Wolfe Algorithm Usage

```julia
# Define your linear optimization oracle
function my_linear_oracle(c::Vector{Float64})
    # Return vertex v of your polytope that minimizes dot(c, v)
    # This is problem-specific - examples:
    # - For base polytope: use greedy algorithm on submodular function
    # - For simplex: return standard basis vector
    # - For flow polytope: solve min-cost flow
    return vertex
end

# Find minimum norm point in polytope
x, iterations, converged = wolfe_algorithm(my_linear_oracle, n; ε=1e-6)
```

### Pre-allocated Version for Performance

```julia
# Create workspace for repeated use
workspace = WolfeWorkspace(n)

# Efficient version with minimal allocations
x, iterations, converged = wolfe_algorithm!(workspace, my_linear_oracle; ε=1e-6)
```

### Example: Minimum Norm Point in Simplex

```julia
using SubmodularMinimization

# Linear oracle for standard simplex {x: x ≥ 0, sum(x) = 1}
function simplex_oracle(c::Vector{Float64})
    # Return vertex e_i where i = argmin c_i
    i_min = argmin(c)
    vertex = zeros(length(c))
    vertex[i_min] = 1.0
    return vertex
end

# Find minimum norm point in simplex
n = 5
x, iters, converged = wolfe_algorithm(simplex_oracle, n)
println("Minimum norm point: $x")
println("Norm: $(norm(x))")
println("Converged in $iters iterations")
```

### Example: Flow Polytope

```julia
# For network flow problems
function flow_oracle(c::Vector{Float64})
    # Solve min-cost flow problem: min c'x subject to Ax = b, x ≥ 0
    # Return extreme point of flow polytope
    # (Implementation depends on your specific network)
    return extreme_flow_point
end

x_min_norm = wolfe_algorithm(flow_oracle, num_edges)[1]
```

### Algorithm Properties

- **Convergence Rate**: O(1/ε) iterations for ε-optimal solution
- **Memory**: O(n²) for storing active vertices (configurable)
- **Numerical Stability**: Built-in safeguards for ill-conditioned problems

This makes the package useful for a much broader class of optimization problems beyond submodular functions.

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
├── test/                                  # Comprehensive test suite (2,980 tests)
└── test_verbose.jl                       # Verbose test runner
```

## Testing

```bash
# Run full test suite (2,980 tests)
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
julia test/runtests.jl      # All 2,980 tests must pass
julia performance_analysis.jl  # Performance regression check
```

## License

GPLv3 License.

---

For detailed algorithmic documentation, implementation details, and theoretical background, see [DETAILS.md](DETAILS.md).