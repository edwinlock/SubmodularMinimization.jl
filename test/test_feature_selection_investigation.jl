"""
Investigation of Feature Selection Function failures.

This file investigates why Feature Selection Functions show 76% success rate
compared to 100% for other function types.
"""

using Test
using SubmodularMinimization
using Random
using Statistics

# Set random seed for reproducibility 
Random.seed!(12345)

@testset "Feature Selection Investigation" begin
    
    println("Investigating Feature Selection Function failures...")
    
    @testset "Basic FeatureSelection Properties" begin
        println("Testing basic properties...")
        
        # Create a simple feature selection function
        n = 4
        relevance = [1.0, 2.0, 3.0, 4.0]
        redundancy = [0.0 0.1 0.2 0.3;
                     0.1 0.0 0.4 0.5;
                     0.2 0.4 0.0 0.6;
                     0.3 0.5 0.6 0.0]
        α = 0.5
        
        f = FeatureSelectionFunction(n, relevance, redundancy, α)
        
        # Test empty set
        empty_set = falses(n)
        @test evaluate(f, empty_set) == 0.0
        
        # Test single elements
        for i in 1:n
            single_set = falses(n)
            single_set[i] = true
            expected = -(α * relevance[i])  # No redundancy for single element
            actual = evaluate(f, single_set)
            @test actual ≈ expected atol=1e-10
            println("  Single element $i: expected=$expected, actual=$actual")
        end
        
        # Test a pair (elements 1 and 2)
        pair_set = BitVector([true, true, false, false])
        expected = -(α * (relevance[1] + relevance[2]) - (1-α) * redundancy[1,2])
        actual = evaluate(f, pair_set)
        @test actual ≈ expected atol=1e-10
        println("  Pair {1,2}: expected=$expected, actual=$actual")
    end
    
    @testset "Numerical Precision Analysis" begin
        println("Analyzing numerical precision issues...")
        
        failures = 0
        differences = Float64[]
        
        for trial in 1:50
            n = rand(4:7)
            f = create_feature_selection(n)
            
            # Use tighter tolerance for main algorithm
            S_main, val_main, _, _ = fujishige_wolfe_submodular_minimization(f; ε=1e-8, verbose=false)
            
            # Brute force
            S_bf, val_bf = brute_force_minimization(f)
            
            diff = abs(val_main - val_bf)
            push!(differences, diff)
            
            # Use a more lenient tolerance for comparison
            tolerance = max(1e-6, 1e-6 * max(abs(val_main), abs(val_bf)))
            value_match = diff <= tolerance
            
            if !value_match
                failures += 1
                println("  FAILURE $failures: n=$n")
                println("    Main: $val_main, Brute force: $val_bf")
                println("    Difference: $diff")
                println("    Tolerance used: $tolerance")
                println("    Main set: $(findall(S_main))")
                println("    BF set: $(findall(S_bf))")
                
                # Let's check the function values manually
                val_main_check = evaluate(f, S_main)
                val_bf_check = evaluate(f, S_bf)
                println("    Manual check - Main: $val_main_check, BF: $val_bf_check")
            end
        end
        
        success_rate = (50 - failures) / 50 * 100
        println("  Success rate: $(round(success_rate, digits=1))% ($failures failures)")
        println("  Difference statistics:")
        println("    Mean: $(round(mean(differences), digits=8))")
        println("    Max: $(round(maximum(differences), digits=8))")
        println("    Std: $(round(std(differences), digits=8))")
        println("    95th percentile: $(round(quantile(differences, 0.95), digits=8))")
    end
    
    @testset "Algorithm Tolerance Sensitivity" begin
        println("Testing sensitivity to algorithm tolerance...")
        
        # Test the same function with different tolerances
        n = 6
        f = create_feature_selection(n)
        
        tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
        results = []
        
        # Get brute force result as ground truth
        S_bf, val_bf = brute_force_minimization(f)
        
        for tol in tolerances
            S_main, val_main, _, iters = fujishige_wolfe_submodular_minimization(f; ε=tol, verbose=false)
            diff = abs(val_main - val_bf)
            push!(results, (tol=tol, val=val_main, diff=diff, iters=iters))
            
            println("  ε=$tol: val=$val_main, diff=$diff, iters=$iters")
        end
        
        # Check if tighter tolerance improves accuracy
        diffs = [r.diff for r in results]
        println("  Differences: $diffs")
        @test diffs[end] <= diffs[1]  # Tighter tolerance should be more accurate
    end
    
    @testset "Specific Failure Case Analysis" begin
        println("Analyzing specific failure cases...")
        
        # Try to reproduce one of the failing cases from the main test
        Random.seed!(42)  # Different seed to potentially hit a failure case
        
        found_failure = false
        for attempt in 1:20
            n = rand(5:7)
            f = create_feature_selection(n)
            
            S_main, val_main, _, _ = fujishige_wolfe_submodular_minimization(f; verbose=false)
            S_bf, val_bf = brute_force_minimization(f)
            
            diff = abs(val_main - val_bf)
            if diff > 1e-6
                found_failure = true
                println("  Found failure case (attempt $attempt):")
                println("    n=$n")
                println("    Main: $val_main, Brute force: $val_bf")
                println("    Difference: $diff")
                println("    Main set: $(findall(S_main)), size=$(sum(S_main))")
                println("    BF set: $(findall(S_bf)), size=$(sum(S_bf))")
                
                # Analyze the function structure
                println("    Function analysis:")
                println("      α=$(f.α)")
                println("      Relevance range: [$(minimum(f.relevance)), $(maximum(f.relevance))]")
                println("      Redundancy range: [$(minimum(f.redundancy)), $(maximum(f.redundancy))]")
                
                # Check if the function is actually submodular
                # Note: check_submodularity is a method, not a field
                # This would need to be implemented separately
                
                # Test different convergence criteria
                S_tight, val_tight, _, _ = fujishige_wolfe_submodular_minimization(f; ε=1e-10, max_iterations=20000, verbose=false)
                diff_tight = abs(val_tight - val_bf)
                println("    With tighter tolerance (ε=1e-10): val=$val_tight, diff=$diff_tight")
                
                break
            end
        end
        
        if !found_failure
            println("  No failure case found in 20 attempts with seed 42")
        end
    end
    
    @testset "Function Value Distribution Analysis" begin
        println("Analyzing function value distributions...")
        
        n = 5
        f = create_feature_selection(n)
        
        # Get all function values using brute force verbose
        S_bf, val_bf, eval_count, all_values = brute_force_minimization_verbose(f; show_progress=false)
        
        println("  Function value statistics (n=$n):")
        println("    Min: $(minimum(all_values))")
        println("    Max: $(maximum(all_values))")
        println("    Mean: $(round(mean(all_values), digits=4))")
        println("    Std: $(round(std(all_values), digits=4))")
        println("    Range: $(maximum(all_values) - minimum(all_values))")
        
        # Check how many values are close to the minimum
        tolerance = 1e-6
        close_to_min = count(v -> abs(v - val_bf) <= tolerance, all_values)
        println("    Values within $tolerance of minimum: $close_to_min / $eval_count")
        
        # Check for multiple local minima
        sorted_values = sort(all_values)
        unique_values = unique(all_values)
        println("    Unique function values: $(length(unique_values)) / $eval_count")
        println("    Smallest 5 values: $(sorted_values[1:min(5, length(sorted_values))])")
    end
    
    println("Feature Selection investigation completed!")
end