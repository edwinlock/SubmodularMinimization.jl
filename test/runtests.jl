"""
Main test runner for SubmodularMinimization.jl

This file runs all test suites following Julia best practices.
"""

using Test
using SubmodularMinimization

@testset "SubmodularMinimization.jl Tests" begin
    include("test_core.jl")
    include("test_examples.jl") 
    include("test_oracles.jl")
    include("test_algorithms.jl")
    include("test_edge_cases.jl")
end