"""
Utility Functions for Testing and Benchmarking

This module contains testing utilities, brute force algorithms for verification,
and benchmarking functions.
"""

using Random

"""
    brute_force_minimization(f::SubmodularFunction)

Brute force minimization for small instances (testing purposes).
Returns (S_min, min_value) where S_min is the minimizing set and min_value is the minimum value.
"""
function brute_force_minimization(f::SubmodularFunction)
    n = ground_set_size(f)
    if n > 20
        error("Brute force only for n ≤ 20 due to computational complexity (2^$n evaluations required)")
    end
    
    min_value = Inf
    S_min = falses(n)
    
    # Iterate through all 2^n possible subsets
    for i in 0:(2^n - 1)
        S = falses(n)
        for j in 1:n
            if (i >> (j-1)) & 1 == 1
                S[j] = true
            end
        end
        
        val = evaluate(f, S)
        if val < min_value
            min_value = val
            S_min = copy(S)
        end
    end
    
    return S_min, min_value
end

"""
    brute_force_minimization_verbose(f::SubmodularFunction; show_progress::Bool=true)

Verbose version of brute force minimization that shows progress and statistics.
Useful for debugging and understanding the search space.
"""
function brute_force_minimization_verbose(f::SubmodularFunction; show_progress::Bool=true)
    n = ground_set_size(f)
    if n > 20
        error("Brute force only for n ≤ 20 due to computational complexity (2^$n evaluations required)")
    end
    
    min_value = Inf
    S_min = falses(n)
    evaluation_count = 0
    values_seen = Float64[]
    
    total_subsets = 2^n
    if show_progress
        println("Brute force search over $total_subsets subsets (n=$n)")
    end
    
    # Iterate through all 2^n possible subsets
    for i in 0:(2^n - 1)
        S = falses(n)
        for j in 1:n
            if (i >> (j-1)) & 1 == 1
                S[j] = true
            end
        end
        
        val = evaluate(f, S)
        evaluation_count += 1
        push!(values_seen, val)
        
        if val < min_value
            min_value = val
            S_min = copy(S)
            if show_progress && evaluation_count % (total_subsets ÷ 10) == 0
                println("  Progress: $(round(100*evaluation_count/total_subsets, digits=1))% - Current best: $min_value")
            end
        end
    end
    
    if show_progress
        println("Brute force complete:")
        println("  Total evaluations: $evaluation_count")
        println("  Minimum value: $min_value")
        println("  Minimizing set size: $(sum(S_min))")
        println("  Value statistics: min=$(minimum(values_seen)), max=$(maximum(values_seen)), mean=$(round(sum(values_seen)/length(values_seen), digits=4))")
    end
    
    return S_min, min_value, evaluation_count, values_seen
end

"""
    test_fujishige_wolfe()

Test the Fujishige-Wolfe algorithm on various submodular functions.
"""
function test_fujishige_wolfe()
    println("Testing Fujishige-Wolfe Algorithm")
    println("=" ^ 40)
    
    # Test 1: Concave submodular function
    println("Test 1: f(S) = |S|^0.7")
    f1 = ConcaveSubmodularFunction(8, 0.7)
    S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization(f1; verbose=true)
    println("Result: |S| = $(sum(S_min)), f(S) = $min_val")
    println("Iterations: $iters")
    
    # Verify with brute force
    S_bf, val_bf = brute_force_minimization(f1)
    println("Brute force: |S| = $(sum(S_bf)), f(S) = $val_bf")
    println("Match: $(abs(min_val - val_bf) < 1e-6)")
    println()
    
    # Test 2: Graph cut function
    println("Test 2: Cut function on random graph")
    Random.seed!(42)
    f2 = create_random_cut_function(6, 0.3)
    println("Graph has $(length(f2.edges)) edges")
    
    S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization(f2; verbose=true)
    cut_indices = findall(S_min)
    println("Result: S = $cut_indices, cut value = $min_val")
    println("Iterations: $iters")
    
    # Verify with brute force
    S_bf, val_bf = brute_force_minimization(f2)
    bf_indices = findall(S_bf)
    println("Brute force: S = $bf_indices, cut value = $val_bf")
    println("Match: $(abs(min_val - val_bf) < 1e-6)")
    println()
    
    # Test 3: Performance test
    println("Test 3: Performance test on larger instance")
    f3 = ConcaveSubmodularFunction(15, 0.8)
    
    time_start = time()
    S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization(f3; ε=1e-4, verbose=false)
    time_elapsed = time() - time_start
    
    println("n=15, f(S) = |S|^0.8")
    println("Result: |S| = $(sum(S_min)), f(S) = $min_val")
    println("Time: $(round(time_elapsed, digits=3)) seconds")
    println("Iterations: $iters")
    
    return S_min, min_val, x, iters
end

"""
    benchmark_implementation(n::Int; num_trials::Int=5)

Benchmark the implementation.
"""
function benchmark_implementation(n::Int; num_trials::Int=5)
    println("Benchmarking Implementation (n=$n)")
    
    # Test concave function
    f1 = ConcaveSubmodularFunction(n, 0.7)
    times = Float64[]
    
    for _ in 1:num_trials
        time_start = time()
        S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization(f1; ε=1e-4, verbose=false)
        elapsed = time() - time_start
        push!(times, elapsed)
    end
    
    avg_time = sum(times) / length(times)
    std_time = sqrt(sum((t - avg_time)^2 for t in times) / (length(times) - 1))
    
    println("Concave function: $(round(avg_time, digits=4)) ± $(round(std_time, digits=4)) seconds")
    
    # Test cut function
    Random.seed!(42)
    f2 = create_random_cut_function(n, 0.2)
    times = Float64[]
    
    for _ in 1:num_trials
        time_start = time()
        S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization(f2; ε=1e-4, verbose=false)
        elapsed = time() - time_start
        push!(times, elapsed)
    end
    
    avg_time = sum(times) / length(times)
    std_time = sqrt(sum((t - avg_time)^2 for t in times) / (length(times) - 1))
    
    println("Cut function: $(round(avg_time, digits=4)) ± $(round(std_time, digits=4)) seconds")
    
    return times
end

"""
    test_implementation()

Test the implementation for correctness.
"""
function test_implementation()
    println("Testing Fujishige-Wolfe Implementation")
    println("=" ^ 50)
    
    # Correctness test
    println("Correctness test (n=8):")
    f = ConcaveSubmodularFunction(8, 0.7)
    S_min, min_val, x, iters = fujishige_wolfe_submodular_minimization(f; verbose=true)
    println()
    
    # Performance tests
    for n in [10, 15, 20, 25]
        benchmark_implementation(n; num_trials=3)
    end
    
    return S_min, min_val
end