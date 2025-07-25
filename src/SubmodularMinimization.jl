"""
SubmodularMinimization.jl

A Julia package for submodular function minimization using the Fujishige-Wolfe algorithm.

This implementation follows the algorithm described in:
"Provable Submodular Minimization using Wolfe's Algorithm"
by Chakrabarty, Jain, and Kothari (2014)

## Main Functions

- `fujishige_wolfe_submodular_minimization`: Main implementation with automatic workspace management
- `fujishige_wolfe_submodular_minimization!`: Direct pre-allocated version for performance
- `wolfe_algorithm`: Core Wolfe's algorithm with automatic workspace management
- `wolfe_algorithm!`: Direct pre-allocated version of core algorithm

## Example Functions

- `ConcaveSubmodularFunction`: f(S) = |S|^α for testing
- `CutFunction`: Graph cut function

## Tolerance System

The package provides standardized tolerance constants for consistent numerical behavior:

- `DEFAULT_TOLERANCE` (1e-6): Balanced accuracy and efficiency for most use cases
- `LOOSE_TOLERANCE` (1e-4): Fast approximate results for prototyping
- `TIGHT_TOLERANCE` (5e-7): Higher precision while remaining practical
- `COMPARISON_TOLERANCE` (1e-10): For comparing algorithm results
- `NUMERICAL_PRECISION_TOLERANCE` (1e-12): Machine epsilon-based limits

## Usage

```julia
using SubmodularMinimization

# Create a test function
f = ConcaveSubmodularFunction(10, 0.7)

# Main algorithm with automatic workspace management (uses DEFAULT_TOLERANCE)
S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization(f; verbose=true)

# Using explicit tolerance constants
S_min_fast, _, _, _ = fujishige_wolfe_submodular_minimization(f; ε=LOOSE_TOLERANCE)
S_min_precise, _, _, _ = fujishige_wolfe_submodular_minimization(f; ε=TIGHT_TOLERANCE)

# Direct pre-allocated version for maximum performance
workspace = WolfeWorkspace(10)  # pre-allocate for ground set size 10
S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization!(
    workspace, f; verbose=true)

# Algorithm with matrix storage optimization
S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization(f; verbose=true)

# Lower-level: just find minimum norm point (without extracting minimizing set)
x, iters, converged = wolfe_algorithm(f; ε=1e-6, verbose=true)

# Pre-allocated version of core algorithm
workspace = WolfeWorkspace(10)
x, iters, converged = wolfe_algorithm!(workspace, f; ε=1e-6)

# Run tests
test_fujishige_wolfe()
```

## Performance Characteristics

For a typical problem (n=10):
- `fujishige_wolfe_submodular_minimization`: ~0.4ms, 106 allocations (auto-managed workspace)
- `fujishige_wolfe_submodular_minimization!`: ~0.4ms, minimal allocations (reused workspace)
- Matrix storage optimization provides ~0.3ms with numerical stability
"""
module SubmodularMinimization

using LinearAlgebra
using Random

# Core interfaces and types
include("core.jl")

# Example submodular functions
include("examples.jl")

# Optimization oracles
include("oracles.jl")

# Main algorithms
include("algorithms.jl")

# Testing and utility functions
include("utils.jl")

# ============================================================================
# TOLERANCE CONSTANTS
# ============================================================================

"""
Numerical tolerances for the Fujishige-Wolfe algorithm.

These constants provide a centralized, well-reasoned tolerance system for
floating-point computations throughout the package.
"""

# Default algorithm convergence tolerance
# This balances accuracy with computational efficiency for most use cases
const DEFAULT_TOLERANCE = 1e-6

# Loose tolerance for quick/approximate results
# Suitable for prototyping, large-scale problems, or when high precision isn't critical
const LOOSE_TOLERANCE = 1e-4

# Tight tolerance for higher-precision results  
# Provides better accuracy while remaining computationally practical
const TIGHT_TOLERANCE = 5e-7

# Comparison tolerance for floating point equality tests
# Used when comparing algorithm results, function values, etc.
const COMPARISON_TOLERANCE = 1e-10

# Numerical precision tolerance (based on machine epsilon)
# Used for detecting when we've reached the limits of floating-point precision
const NUMERICAL_PRECISION_TOLERANCE = 1e-12

# ============================================================================

# Export core types and interfaces
export SubmodularFunction, evaluate, ground_set_size

# Export example functions
export ConcaveSubmodularFunction, CutFunction, create_random_cut_function
export MatroidRankFunction, FacilityLocationFunction, LogDeterminantFunction  
export EntropyFunction, SquareRootFunction, WeightedCoverageFunction
export BipartiteMatchingFunction, AsymmetricCutFunction, ConcaveWithPenalty

# Export AI and image recognition functions
export ImageSegmentationFunction, FeatureSelectionFunction, DiversityFunction
export SensorPlacementFunction, InformationGainFunction

# Export economics and auction theory functions
export GrossSubstitutesFunction, AuctionRevenueFunction, MarketShareFunction

# Export helper constructors for random instances
export create_random_facility_location, create_random_coverage_function
export create_wishart_log_determinant, create_random_entropy_function
export create_bipartite_matching, create_asymmetric_cut, create_concave_with_penalty

# Export AI/vision and economics constructors
export create_image_segmentation, create_feature_selection, create_diversity_function
export create_sensor_placement, create_information_gain
export create_gross_substitutes, create_auction_revenue, create_market_share

# Export main algorithms
export linear_optimization_oracle, affine_minimizer
export wolfe_algorithm, fujishige_wolfe_submodular_minimization

# Export pre-allocated versions for maximum performance
export WolfeWorkspace, AffineWorkspace
export linear_optimization_oracle!, affine_minimizer!
export wolfe_algorithm!, fujishige_wolfe_submodular_minimization!

# Export utilities
export brute_force_minimization, brute_force_minimization_verbose, test_fujishige_wolfe
export benchmark_implementation, test_implementation

# Export tolerance constants
export DEFAULT_TOLERANCE, LOOSE_TOLERANCE, TIGHT_TOLERANCE, COMPARISON_TOLERANCE, NUMERICAL_PRECISION_TOLERANCE

end # module SubmodularMinimization