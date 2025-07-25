#!/usr/bin/env julia
"""
Profiling Script for Fujishige-Wolfe Implementation

This script runs the implementation on a single example submodular 
function with sufficient work to enable meaningful profiling with @profview.

Usage:
    julia profile_standard.jl

For profiling:
    julia> using ProfileView
    julia> include("profile_standard.jl")
    julia> @profview profile_example()

The function is designed to be computationally intensive enough to show
meaningful profiling data while being representative of typical usage.
"""

using Pkg
Pkg.activate(".")
using SubmodularMinimization
using LinearAlgebra
using Printf
using Statistics

"""
Profile a single computationally intensive example
"""
function profile_example()
    println("🔍 Starting profiling example...")
    
    # Create a large problem that should take around 1 second to converge
    # Based on performance analysis, concave functions scale better than facility location
    # n=25 takes 0.112ms with 30 iterations, so we need ~100x scale-up
    
    # Large concave function with penalty to ensure non-trivial minimizer
    n = 500  # Very large problem size
    α = 0.85  # Higher α for more complexity
    penalty = 15.0
    
    println("📊 Problem: Large ConcaveWithPenalty(n=$n, α=$α, penalty=$penalty)")
    println("⚙️  This should take around 1 second to converge...")
    
    # Create the function
    f = create_concave_with_penalty(n, α, penalty)
    
    # Time the execution
    start_time = time()
    
    # Run the algorithm with detailed output
    S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(
        f; 
        ε=1e-6,  # Use default tolerance to avoid convergence issues
        max_iterations=5000,
        verbose=true
    )
    
    end_time = time()
    elapsed = end_time - start_time
    
    println("\n📈 Results:")
    @printf "  Execution time: %.3f seconds\n" elapsed
    @printf "  Iterations: %d\n" iterations
    @printf "  Minimum value: %.8f\n" min_val
    @printf "  Solution norm: %.8f\n" norm(x)
    @printf "  Minimizing set size: %d\n" sum(S_min)
    @printf "  Time per iteration: %.3f ms\n" (elapsed * 1000 / iterations)
    
    if sum(S_min) > 0
        selected_indices = findall(S_min)
        println("  Selected elements: $(length(selected_indices)) out of $n")
        if length(selected_indices) <= 10
            println("  Element indices: $selected_indices")
        else
            println("  First 10 elements: $(selected_indices[1:10])...")
        end
    else
        println("  Minimizing set: ∅ (empty set)")
    end
    
    return S_min, min_val, x, iterations
end

"""
Profile multiple runs for statistical analysis
"""
function profile_multiple_runs(num_runs::Int=5)
    println("🔄 Running $num_runs iterations for statistical profiling...")
    
    times = Float64[]
    iterations_list = Int[]
    
    # Mix of complex problems that showed high iteration counts
    problem_types = [
        :facility_location,
        :concave_penalty,
        :bipartite_matching,
        :asymmetric_cut,
        :weighted_coverage
    ]
    
    for i in 1:num_runs
        println("\n--- Run $i/$num_runs ---")
        
        # Cycle through different complex problem types
        problem_type = problem_types[(i-1) % length(problem_types) + 1]
        
        # Create different problems based on type
        if problem_type == :facility_location
            f = create_random_facility_location(4 + i, 6 + i; max_weight=10.0)
            desc = "FacilityLocation($(4+i),$(6+i))"
        elseif problem_type == :concave_penalty
            f = create_concave_with_penalty(6 + i, 0.6, 2.0 + i*0.5)
            desc = "ConcaveWithPenalty($(6+i))"
        elseif problem_type == :bipartite_matching
            f = create_bipartite_matching(3 + i, 3 + i, 0.4 + i*0.1)
            desc = "BipartiteMatching($(3+i)+$(3+i))"
        elseif problem_type == :asymmetric_cut
            f = create_asymmetric_cut(8 + i, 0.4, 2.0 + i*0.5)
            desc = "AsymmetricCut($(8+i))"
        else  # weighted_coverage
            f = create_random_coverage_function(6 + i, 8 + i)
            desc = "WeightedCoverage($(6+i),$(8+i))"
        end
        
        # Time the execution
        start_time = time()
        S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(
            f; 
            ε=1e-7,
            max_iterations=3000,
            verbose=false  # Suppress output for cleaner profiling
        )
        end_time = time()
        
        elapsed = end_time - start_time
        push!(times, elapsed)
        push!(iterations_list, iterations)
        
        @printf "  Problem: %s → %.3f seconds, %d iterations\n" desc elapsed iterations
    end
    
    println("\n📊 Summary Statistics:")
    @printf "  Average time: %.3f ± %.3f seconds\n" mean(times) std(times)
    @printf "  Average iterations: %.1f ± %.1f\n" mean(iterations_list) std(iterations_list)
    @printf "  Total time: %.3f seconds\n" sum(times)
    
    return times, iterations_list
end

"""
Profile cut function (different algorithmic characteristics)
"""
function profile_cut_function()
    println("🔗 Profiling complex cut function example...")
    
    # Use asymmetric cut which showed more complexity in performance analysis
    n = 15
    f = create_asymmetric_cut(n, 0.6, 5.0)  # Higher density and asymmetry
    
    println("📊 Problem: AsymmetricCutFunction(n=$n, asymmetry=5.0)")
    println("⚙️  This uses asymmetric penalties to create more complex optimization...")
    
    # Run the algorithm
    S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(
        f;
        ε=1e-7,
        max_iterations=3000,
        verbose=true
    )
    
    println("\n📈 Results:")
    @printf "  Iterations: %d\n" iterations
    @printf "  Minimum value: %.8f\n" min_val
    @printf "  Solution norm: %.8f\n" norm(x)
    @printf "  Cut size: %d\n" Int(min_val)
    @printf "  Minimizing set size: %d\n" sum(S_min)
    
    return S_min, min_val, x, iterations
end

"""
Comprehensive profiling session
"""
function comprehensive_profile()
    println("🎯 Comprehensive Profiling Session")
    println("="^50)
    
    println("\n1️⃣ Single intensive run (best for @profview):")
    profile_example()
    
    println("\n\n2️⃣ Multiple runs (statistical analysis):")
    profile_multiple_runs(3)
    
    println("\n\n3️⃣ Cut function profiling:")
    profile_cut_function()
    
    println("\n✅ Profiling session complete!")
    println("\n💡 To profile with ProfileView:")
    println("   julia> using ProfileView")
    println("   julia> include(\"profile_standard.jl\")")
    println("   julia> @profview profile_example()  # or comprehensive_profile()")
end

# If run directly, do a single profiling example
if abspath(PROGRAM_FILE) == @__FILE__
    profile_example()
end

# Functions available for interactive profiling:
# profile_example() - Single intensive run
# profile_multiple_runs() - Multiple statistical runs
# profile_cut_function() - Cut function specific
# comprehensive_profile() - All of the above