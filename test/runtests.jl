"""
Main test runner for SubmodularMinimization.jl

This file runs all test suites following Julia best practices.
"""

using Test
using SubmodularMinimization
using Statistics

@testset "SubmodularMinimization.jl Tests" begin
    # Core functionality tests
    include("test_core.jl")
    include("test_examples.jl") 
    include("test_oracles.jl")
    include("test_algorithms.jl")
    include("test_edge_cases.jl")
    
    # Utility function tests
    include("test_is_submodular.jl")
    include("test_is_minimiser.jl")
    
    # Algorithm verification tests
    include("test_algorithm_verification.jl")
    include("test_brute_force_simple.jl")
    include("test_extensive_randomized_correctness.jl")
end