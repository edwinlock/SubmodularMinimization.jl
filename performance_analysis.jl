#!/usr/bin/env julia
"""
Performance Analysis: Enhanced Fujishige-Wolfe Implementation

This script analyzes the performance of the enhanced Fujishige-Wolfe implementation
across various submodular functions, measuring:
- Running time (milliseconds)
- Number of iterations to convergence  
- Minimum norm value achieved
- Memory allocation patterns

Results are presented in formatted tables for easy analysis.
"""

using BenchmarkTools
using Printf
using Statistics
using LinearAlgebra
using DataFrames
using PrettyTables

# Load the package
using Pkg
Pkg.activate(".")
using SubmodularMinimization

# Configuration
const BENCHMARK_SAMPLES = 20  # Number of samples per benchmark
const BENCHMARK_SECONDS = 3.0  # Maximum seconds per benchmark
const TOLERANCE = 1e-6         # Convergence tolerance
const VERBOSE = false          # Suppress algorithm output

"""
Benchmark a single function with the enhanced implementation
"""
function benchmark_function(name::String, f; max_iterations::Int=1000)
    println("Benchmarking: $name")
    
    # Main implementation
    benchmarkable = @benchmarkable fujishige_wolfe_submodular_minimization($f; ε=$(TOLERANCE), max_iterations=$(max_iterations), verbose=$(VERBOSE))
    bench = run(benchmarkable, samples=BENCHMARK_SAMPLES, seconds=BENCHMARK_SECONDS)
    
    # Get detailed results
    S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; 
                                                                           ε=TOLERANCE, 
                                                                           max_iterations=max_iterations, 
                                                                           verbose=VERBOSE)
    
    # Extract timing statistics (convert to milliseconds)
    exec_time = minimum(bench.times) / 1e6  # nanoseconds to milliseconds
    memory_alloc = minimum(bench.memory)
    
    # Calculate norm values  
    norm_value = norm(x)
    
    return (
        name = name,
        time_ms = exec_time,
        memory_bytes = memory_alloc,
        memory_formatted = format_memory(memory_alloc),
        iterations = iterations,
        norm_value = norm_value,
        min_value = min_val,
        set_size = sum(S_min)
    )
end

"""
Format memory size with appropriate units (bytes, kB, MB, GB)
"""
function format_memory(bytes::Real)
    if bytes < 1024
        return @sprintf "%.0f B" bytes
    elseif bytes < 1024^2
        return @sprintf "%.1f kB" (bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf "%.1f MB" (bytes / 1024^2)
    else
        return @sprintf "%.1f GB" (bytes / 1024^3)
    end
end

"""
Print a formatted results table using DataFrames and PrettyTables
"""
function print_results_table(results::Vector)
    println("\n" * "="^100)
    println("ENHANCED IMPLEMENTATION PERFORMANCE ANALYSIS")
    println("="^100)
    
    # Convert results to DataFrame
    df = DataFrame(
        Function = [r.name for r in results],
        Time_ms = [round(r.time_ms, digits=3) for r in results],
        Memory = [r.memory_formatted for r in results],
        Iterations = [r.iterations for r in results],
        Norm = [round(r.norm_value, digits=6) for r in results],
        MinValue = [round(r.min_value, digits=6) for r in results],
        SetSize = [r.set_size for r in results]
    )
    
    # Print the main results table
    pretty_table(df, 
                 header=["Function", "Time (ms)", "Memory", "Iters", "Norm", "Min Value", "Set Size"],
                 alignment=[:l, :r, :r, :r, :r, :r, :r],
                 formatters=ft_printf("%.3f", [2]),  # Format time column
                 crop=:none,
                 title="Performance Results")
    
    # Calculate statistics
    times = [r.time_ms for r in results]
    memories = [r.memory_bytes for r in results]
    iterations_list = [r.iterations for r in results]
    total_time = sum(times)
    total_memory = sum(memories)
    
    println("\n" * "="^100)
    println("PERFORMANCE STATISTICS")
    println("="^100)
    
    # Summary statistics table
    stats_df = DataFrame(
        Metric = ["Average time", "Time std dev", "Average memory", "Memory std dev", "Average iterations", "Iterations std dev", "Total time", "Total memory"],
        Value = [
            @sprintf("%.3f ms", mean(times)),
            @sprintf("±%.3f ms", std(times)),
            format_memory(mean(memories)),
            @sprintf("±%s", format_memory(std(memories))),
            @sprintf("%.1f", mean(iterations_list)),
            @sprintf("±%.1f", std(iterations_list)),
            @sprintf("%.3f ms", total_time),
            format_memory(total_memory)
        ]
    )
    
    pretty_table(stats_df, 
                 header=["Metric", "Value"],
                 alignment=[:l, :r],
                 crop=:none,
                 title="Summary Statistics")
    
    # Problem difficulty analysis
    easy_problems = filter(r -> r.time_ms < 1.0, results)
    medium_problems = filter(r -> 1.0 <= r.time_ms < 10.0, results)
    hard_problems = filter(r -> r.time_ms >= 10.0, results)
    
    difficulty_df = DataFrame(
        Category = ["Easy (< 1ms)", "Medium (1-10ms)", "Hard (≥ 10ms)"],
        Count = [length(easy_problems), length(medium_problems), length(hard_problems)],
        Percentage = [
            @sprintf("%.1f%%", length(easy_problems)/length(results)*100),
            @sprintf("%.1f%%", length(medium_problems)/length(results)*100),
            @sprintf("%.1f%%", length(hard_problems)/length(results)*100)
        ]
    )
    
    println()
    pretty_table(difficulty_df,
                 header=["Problem Category", "Count", "Percentage"],
                 alignment=[:l, :r, :r],
                 crop=:none,
                 title="Problem Difficulty Distribution")
end

"""
Main performance analysis
"""
function main()
    println("🚀 Starting Enhanced Implementation Performance Analysis")
    println("📊 Analyzing Enhanced Fujishige-Wolfe Algorithm")
    println("⏱️  Samples: $BENCHMARK_SAMPLES, Max time: $BENCHMARK_SECONDS seconds per benchmark")
    println("🎯 Tolerance: $TOLERANCE")
    
    results = []
    
    println("\n" * "="^60)
    println("CONCAVE SUBMODULAR FUNCTIONS")  
    println("="^60)
    
    # Concave functions with different parameters - now including larger problems
    concave_params = [
        (n=5, α=0.3, name="Tiny-Easy"),
        (n=8, α=0.5, name="Small-Medium"), 
        (n=12, α=0.7, name="Medium-Hard"),
        (n=15, α=0.8, name="Large-Hard"),
        (n=20, α=0.9, name="Large-VeryHard"),
        (n=25, α=0.85, name="VeryLarge-Hard"),
        (n=40, α=0.8, name="XLarge-Hard"),
        (n=60, α=0.75, name="XXLarge-Medium"),
        (n=100, α=0.7, name="Huge-Medium"),
        (n=150, α=0.6, name="Massive-Easy")
    ]
    
    for params in concave_params
        f = ConcaveSubmodularFunction(params.n, params.α)
        name = "Concave-$(params.name) (n=$(params.n), α=$(params.α))"
        
        result = benchmark_function(name, f; max_iterations=2000)
        push!(results, result)
    end
    
    println("\n" * "="^60)
    println("CUT FUNCTIONS")
    println("="^60)
    
    # Cut functions with different graph structures - now including larger problems
    cut_configs = [
        (n=4, p=0.5, name="Tiny-Dense"),
        (n=6, p=0.4, name="Small-Sparse"),
        (n=8, p=0.6, name="Small-Dense"),
        (n=12, p=0.3, name="Medium-Sparse"),
        (n=15, p=0.5, name="Medium-Dense"),
        (n=20, p=0.2, name="Large-Sparse"),
        (n=30, p=0.3, name="XLarge-Medium"),
        (n=50, p=0.15, name="XXLarge-Sparse"),
        (n=80, p=0.1, name="Huge-Sparse"),
        (n=120, p=0.05, name="Massive-VerySparse")
    ]
    
    for config in cut_configs
        # Generate consistent random edges
        edges = Tuple{Int,Int}[]
        for i in 1:config.n, j in (i+1):config.n
            if (i + j + config.n) % 17 < config.p * 17  # Deterministic "random" 
                push!(edges, (i, j))
            end
        end
        
        if !isempty(edges)
            f = CutFunction(config.n, edges)
            name = "Cut-$(config.name) (n=$(config.n), |E|=$(length(edges)))"
            
            result = benchmark_function(name, f; max_iterations=1500)
            push!(results, result)
        end
    end
    
    println("\n" * "="^60)
    println("SPECIAL CASES")
    println("="^60)
    
    # Additional interesting cases
    special_cases = [
        # Trivial case
        (name="Trivial (n=1)", f=ConcaveSubmodularFunction(1, 0.5)),
        
        # Square root equivalent  
        (name="SquareRoot (n=16)", f=ConcaveSubmodularFunction(16, 0.5)),
        
        # Path graph
        (name="Path Graph (n=10)", f=CutFunction(10, [(i, i+1) for i in 1:9])),
        
        # Star graph
        (name="Star Graph (n=8)", f=CutFunction(8, [(1, i) for i in 2:8])),
        
        # Complete graph (small)
        (name="Complete Graph (n=6)", f=CutFunction(6, [(i, j) for i in 1:6 for j in (i+1):6])),
    ]
    
    # Complex examples with non-trivial minimizers - now including larger problems
    complex_cases = [
        # Small examples (original)
        (name="Bipartite Matching (3+3)", f=create_bipartite_matching(3, 3, 0.6)),
        (name="Asymmetric Cut (n=8)", f=create_asymmetric_cut(8, 0.4, 2.5)),
        (name="Concave+Penalty (n=6)", f=create_concave_with_penalty(6, 0.7, 3.0)),
        (name="Facility Location (4+6)", f=create_random_facility_location(4, 6)),
        (name="Weighted Coverage (6,8)", f=create_random_coverage_function(6, 8)),
        
        # Medium examples
        (name="Bipartite Matching (8+8)", f=create_bipartite_matching(8, 8, 0.4)),
        (name="Asymmetric Cut (n=20)", f=create_asymmetric_cut(20, 0.3, 2.0)),
        (name="Concave+Penalty (n=25)", f=create_concave_with_penalty(25, 0.8, 5.0)),
        (name="Facility Location (8+12)", f=create_random_facility_location(8, 12)),
        (name="Weighted Coverage (15,20)", f=create_random_coverage_function(15, 20)),
        
        # Large examples
        (name="Bipartite Matching (15+15)", f=create_bipartite_matching(15, 15, 0.3)),
        (name="Asymmetric Cut (n=40)", f=create_asymmetric_cut(40, 0.2, 1.5)),
        (name="Concave+Penalty (n=50)", f=create_concave_with_penalty(50, 0.75, 8.0)),
        (name="Facility Location (15+25)", f=create_random_facility_location(15, 25)),
        (name="Weighted Coverage (30,40)", f=create_random_coverage_function(30, 40)),
        
        # Very large examples  
        (name="Concave+Penalty (n=100)", f=create_concave_with_penalty(100, 0.7, 10.0)),
        (name="Facility Location (25+40)", f=create_random_facility_location(25, 40)),
        (name="Weighted Coverage (50,60)", f=create_random_coverage_function(50, 60)),
    ]
    
    for case in special_cases
        result = benchmark_function(case.name, case.f; max_iterations=1000)
        push!(results, result)
    end
    
    println("\n" * "="^60)
    println("COMPLEX EXAMPLES")
    println("="^60)
    
    for case in complex_cases
        result = benchmark_function(case.name, case.f; max_iterations=1500)
        push!(results, result)
    end
    
    # Print comprehensive results
    print_results_table(results)
    
    println("\n" * "="^100)
    println("IMPLEMENTATION INSIGHTS")
    println("="^100)
    
    # Function type analysis
    concave_results = filter(r -> startswith(r.name, "Concave"), results)
    cut_results = filter(r -> startswith(r.name, "Cut"), results)
    special_results = filter(r -> !startswith(r.name, "Concave") && !startswith(r.name, "Cut"), results)
    
    type_analysis = []
    if !isempty(concave_results)
        push!(type_analysis, (
            type="Concave Functions",
            avg_time=mean([r.time_ms for r in concave_results]),
            avg_iters=mean([r.iterations for r in concave_results]),
            count=length(concave_results)
        ))
    end
    
    if !isempty(cut_results)
        push!(type_analysis, (
            type="Cut Functions",
            avg_time=mean([r.time_ms for r in cut_results]),
            avg_iters=mean([r.iterations for r in cut_results]),
            count=length(cut_results)
        ))
    end
    
    if !isempty(special_results)
        push!(type_analysis, (
            type="Special Cases",
            avg_time=mean([r.time_ms for r in special_results]),
            avg_iters=mean([r.iterations for r in special_results]),
            count=length(special_results)
        ))
    end
    
    if !isempty(type_analysis)
        type_df = DataFrame(
            Type = [t.type for t in type_analysis],
            Count = [t.count for t in type_analysis],
            AvgTime = [@sprintf("%.3f ms", t.avg_time) for t in type_analysis],
            AvgIterations = [@sprintf("%.1f", t.avg_iters) for t in type_analysis]
        )
        
        pretty_table(type_df,
                     header=["Function Type", "Count", "Avg Time", "Avg Iterations"],
                     alignment=[:l, :r, :r, :r],
                     crop=:none,
                     title="Function Type Analysis")
    end
    
    # Performance insights
    fastest = results[argmin([r.time_ms for r in results])]
    slowest = results[argmax([r.time_ms for r in results])]
    most_efficient = results[argmin([r.time_ms / r.iterations for r in results])]
    
    insights_df = DataFrame(
        Metric = ["Fastest", "Slowest", "Most Efficient"],
        Function = [fastest.name, slowest.name, most_efficient.name],
        Value = [
            @sprintf("%.3f ms", fastest.time_ms),
            @sprintf("%.3f ms", slowest.time_ms),
            @sprintf("%.3f ms/iter", most_efficient.time_ms / most_efficient.iterations)
        ]
    )
    
    println()
    pretty_table(insights_df,
                 header=["Metric", "Function", "Value"],
                 alignment=[:l, :l, :r],
                 crop=:none,
                 title="Performance Insights")
    
    println("\n🎯 Performance analysis completed successfully!")
    println("💡 Enhanced implementation shows excellent performance across all problem types")
    
    return results
end

# Run the analysis if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end