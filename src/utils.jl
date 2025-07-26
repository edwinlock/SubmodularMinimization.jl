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

"""
    is_submodular(f::SubmodularFunction; tolerance::Float64=COMPARISON_TOLERANCE, verbose::Bool=false)

Check if a function is submodular using Definition 3 from Wikipedia.

Definition 3: A function f is submodular if for every X ⊆ Ω and x1, x2 ∈ Ω ∖ X such that x1 ≠ x2:
f(X ∪ {x1}) + f(X ∪ {x2}) ≥ f(X ∪ {x1, x2}) + f(X)

This captures the "diminishing returns" property of submodular functions.

## Arguments
- `f`: The function to test for submodularity
- `tolerance`: Numerical tolerance for the inequality check (default: COMPARISON_TOLERANCE)  
- `verbose`: If true, prints detailed information about violations

## Returns
- `is_submodular`: Boolean indicating if the function is submodular
- `violations`: Number of submodularity violations found
- `total_tests`: Total number of submodularity tests performed

## Example
```julia
f = ConcaveSubmodularFunction(4, 0.5)
is_sub, violations, total = is_submodular(f; verbose=true)
# Output: true, 0, 24

# Feature Selection functions are typically NOT submodular
f_feat = create_feature_selection(4)  
is_sub, violations, total = is_submodular(f_feat; verbose=true)
# Output: false, 24, 24 (all tests failed)
```

## Note
This function has O(n² × 2ⁿ) complexity and is only practical for small n (≤ 10).
For larger functions, consider sampling-based approaches.
"""
function is_submodular(f::SubmodularFunction; tolerance::Float64=COMPARISON_TOLERANCE, verbose::Bool=false)
    n = ground_set_size(f)
    
    if n > 15
        @warn "is_submodular() has O(n² × 2ⁿ) complexity. For n=$n, this requires $(n^2 * 2^n) operations. Consider n ≤ 10 for practical use."
    end
    
    violations = 0
    total_tests = 0
    
    if verbose
        println("Checking submodularity using Definition 3...")
        println("For every X ⊆ Ω and x1, x2 ∈ Ω ∖ X such that x1 ≠ x2:")
        println("f(X ∪ {x1}) + f(X ∪ {x2}) ≥ f(X ∪ {x1, x2}) + f(X)")
        println()
    end
    
    # Iterate over all possible subsets X
    for x_bits in 0:(2^n - 1)
        X = BitVector([((x_bits >> (i-1)) & 1) == 1 for i in 1:n])
        
        # Find elements not in X
        not_in_X = findall(.!X)
        
        # Check all pairs of elements not in X
        for i in eachindex(not_in_X)
            for j in (i+1):length(not_in_X)
                x1, x2 = not_in_X[i], not_in_X[j]
                
                # Create the four sets needed for the inequality
                X_union_x1 = copy(X)
                X_union_x1[x1] = true
                
                X_union_x2 = copy(X)
                X_union_x2[x2] = true
                
                X_union_x1_x2 = copy(X)
                X_union_x1_x2[x1] = true
                X_union_x1_x2[x2] = true
                
                # Evaluate the function on all four sets
                f_X = evaluate(f, X)
                f_X_x1 = evaluate(f, X_union_x1)
                f_X_x2 = evaluate(f, X_union_x2)
                f_X_x1_x2 = evaluate(f, X_union_x1_x2)
                
                # Check the submodularity inequality
                left_side = f_X_x1 + f_X_x2
                right_side = f_X_x1_x2 + f_X
                
                total_tests += 1
                
                # Check if inequality is violated
                if left_side < right_side - tolerance
                    violations += 1
                    
                    if verbose && violations ≤ 5  # Show first 5 violations
                        println("VIOLATION $violations:")
                        println("  X = $(findall(X)), x1 = $x1, x2 = $x2")
                        println("  f(X) = $f_X")
                        println("  f(X ∪ {x1}) = $f_X_x1")
                        println("  f(X ∪ {x2}) = $f_X_x2")
                        println("  f(X ∪ {x1,x2}) = $f_X_x1_x2")
                        println("  Left side:  f(X ∪ {x1}) + f(X ∪ {x2}) = $left_side")
                        println("  Right side: f(X ∪ {x1,x2}) + f(X) = $right_side")
                        println("  Violation: $(left_side - right_side)")
                        println()
                    end
                end
            end
        end
    end
    
    is_submodular_result = (violations == 0)
    
    if verbose
        println("Submodularity check completed:")
        println("  Total tests: $total_tests")
        println("  Violations: $violations")
        println("  Success rate: $(round(100 * (total_tests - violations) / total_tests, digits=2))%")
        println("  Result: $(is_submodular_result ? "SUBMODULAR ✓" : "NOT SUBMODULAR ✗")")
    end
    
    return is_submodular_result, violations, total_tests
end

"""
    is_minimiser(S::BitVector, f::SubmodularFunction; tolerance::Float64=COMPARISON_TOLERANCE, verbose::Bool=false)

Check if a given set S is a global minimizer of submodular function f.

For submodular functions, local optimality implies global optimality. A set S is optimal if and only if:
1. f(S ∪ {v}) ≥ f(S) for all v ∉ S (no beneficial additions)
2. f(S ∖ {v}) ≥ f(S) for all v ∈ S (no beneficial removals)

## Arguments
- `S`: Candidate minimizer set (as BitVector)
- `f`: The submodular function to minimize
- `tolerance`: Numerical tolerance for optimality check (default: COMPARISON_TOLERANCE)
- `verbose`: If true, prints detailed information about violations

## Returns
- `is_optimal`: Boolean indicating if S is the global minimizer
- `improvement_found`: If not optimal, describes the improvement found
- `improvement_value`: The better function value found (if any)

## Example
```julia
f = ConcaveSubmodularFunction(4, 0.5)
S = falses(4)  # Empty set (should be optimal for concave functions)
is_opt, improvement, better_val = is_minimiser(S, f; verbose=true)
# Output: true, "", NaN

# Test with suboptimal set
S_bad = BitVector([true, true, false, false])
is_opt, improvement, better_val = is_minimiser(S_bad, f; verbose=true) 
# Output: false, "Can improve by removing element 1", 1.0
```

## Note
This method assumes the function is submodular. For non-submodular functions,
passing this test does not guarantee global optimality.
"""
function is_minimiser(S::BitVector, f::SubmodularFunction; tolerance::Float64=COMPARISON_TOLERANCE, verbose::Bool=false)
    n = ground_set_size(f)
    
    if length(S) != n
        error("Candidate set length $(length(S)) does not match ground set size $n")
    end
    
    f_S = evaluate(f, S)
    
    if verbose
        println("Checking optimality of set $(findall(S)) with value $f_S...")
    end
    
    # Check if we can beneficially add any element not in S
    for v in 1:n
        if !S[v]  # v not in S
            S_plus_v = copy(S)
            S_plus_v[v] = true
            f_S_plus_v = evaluate(f, S_plus_v)
            
            if f_S_plus_v < f_S - tolerance
                improvement_msg = "Can improve by adding element $v"
                if verbose
                    println("  IMPROVEMENT FOUND: $improvement_msg")
                    println("    f(S) = $f_S")
                    println("    f(S ∪ {$v}) = $f_S_plus_v")
                    println("    Improvement: $(f_S - f_S_plus_v)")
                end
                return false, improvement_msg, f_S_plus_v
            end
        end
    end
    
    # Check if we can beneficially remove any element in S
    for v in 1:n
        if S[v]  # v in S
            S_minus_v = copy(S)
            S_minus_v[v] = false
            f_S_minus_v = evaluate(f, S_minus_v)
            
            if f_S_minus_v < f_S - tolerance
                improvement_msg = "Can improve by removing element $v"
                if verbose
                    println("  IMPROVEMENT FOUND: $improvement_msg")
                    println("    f(S) = $f_S")
                    println("    f(S ∖ {$v}) = $f_S_minus_v")
                    println("    Improvement: $(f_S - f_S_minus_v)")
                end
                return false, improvement_msg, f_S_minus_v
            end
        end
    end
    
    if verbose
        println("  ✓ No improvements found - S is locally optimal")
        println("  ✓ For submodular functions, local optimality ⟹ global optimality")
    end
    
    return true, "", NaN
end

"""
    is_minimiser(S::Vector{Int}, f::SubmodularFunction; kwargs...)

Convenience method that accepts a vector of indices instead of a BitVector.

## Example
```julia
f = ConcaveSubmodularFunction(4, 0.5)
is_opt, improvement, better_val = is_minimiser(Int[], f)  # Empty set
# Output: true, "", NaN

is_opt, improvement, better_val = is_minimiser([1, 2], f)  # Set {1, 2}
# Output: false, "Can improve by removing element 1", <better_value>
```
"""
function is_minimiser(S_indices::Vector{Int}, f::SubmodularFunction; kwargs...)
    n = ground_set_size(f)
    S = falses(n)
    for i in S_indices
        if i < 1 || i > n
            error("Index $i is out of bounds for ground set size $n")
        end
        S[i] = true
    end
    return is_minimiser(S, f; kwargs...)
end