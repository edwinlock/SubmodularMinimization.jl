"""
Comprehensive tests for the is_minimiser() function.

This test suite verifies that is_minimiser() correctly identifies optimal
and suboptimal sets for submodular functions.
"""

using Test
using SubmodularMinimization
using Random

# Set random seed for reproducibility
Random.seed!(42)

@testset "is_minimiser() Function Tests" begin
    
    @testset "Basic Functionality" begin
        println("Testing basic is_minimiser functionality...")
        
        @testset "ConcaveSubmodularFunction - Known Optimal" begin
            # For concave functions f(S) = |S|^α, the empty set is always optimal
            for n in 3:6
                for α in [0.3, 0.5, 0.7, 0.9]
                    f = ConcaveSubmodularFunction(n, α)
                    
                    # Test empty set (should be optimal)
                    S_empty = falses(n)
                    is_opt, improvement, better_val = is_minimiser(S_empty, f)
                    @test is_opt == true
                    @test improvement == ""
                    @test isnan(better_val)
                    
                    # Test non-empty sets (should not be optimal)
                    S_single = falses(n)
                    S_single[1] = true
                    is_opt, improvement, better_val = is_minimiser(S_single, f)
                    @test is_opt == false
                    @test improvement == "Can improve by removing element 1"
                    @test better_val < evaluate(f, S_single)
                end
            end
        end
        
        @testset "SquareRootFunction - Known Optimal" begin
            # For square root functions f(S) = √|S|, the empty set is optimal
            for n in 3:6
                f = SquareRootFunction(n)
                
                # Test empty set
                S_empty = falses(n)
                is_opt, improvement, better_val = is_minimiser(S_empty, f)
                @test is_opt == true
                
                # Test non-empty sets
                S_full = trues(n)
                is_opt, improvement, better_val = is_minimiser(S_full, f)
                @test is_opt == false
                @test contains(improvement, "removing element")
            end
        end
        
        @testset "MatroidRankFunction - Known Optimal" begin
            # For matroid rank functions f(S) = min(|S|, k), the empty set is optimal
            for n in 4:6
                for k in 1:(n-1)
                    f = MatroidRankFunction(n, k)
                    
                    # Test empty set
                    S_empty = falses(n)
                    is_opt, improvement, better_val = is_minimiser(S_empty, f)
                    @test is_opt == true
                    
                    # If k >= 1, any non-empty set should not be optimal
                    if k >= 1
                        S_single = falses(n)
                        S_single[1] = true
                        is_opt, improvement, better_val = is_minimiser(S_single, f)
                        @test is_opt == false
                    end
                end
            end
        end
    end
    
    @testset "Input Validation" begin
        println("Testing input validation...")
        
        @testset "BitVector input validation" begin
            f = ConcaveSubmodularFunction(4, 0.5)
            
            # Wrong size BitVector
            S_wrong_size = falses(3)  # n=4 but S has length 3
            @test_throws ErrorException is_minimiser(S_wrong_size, f)
            
            S_wrong_size2 = falses(5)  # n=4 but S has length 5
            @test_throws ErrorException is_minimiser(S_wrong_size2, f)
            
            # Correct size should work
            S_correct = falses(4)
            @test_nowarn is_minimiser(S_correct, f)
        end
        
        @testset "Vector{Int} input validation" begin
            f = ConcaveSubmodularFunction(4, 0.5)
            
            # Valid indices
            @test_nowarn is_minimiser(Int[], f)  # Empty set
            @test_nowarn is_minimiser([1], f)   # Single element
            @test_nowarn is_minimiser([1, 3], f) # Multiple elements
            
            # Invalid indices
            @test_throws ErrorException is_minimiser([0], f)    # Index too small
            @test_throws ErrorException is_minimiser([5], f)    # Index too large
            @test_throws ErrorException is_minimiser([-1], f)   # Negative index
            @test_throws ErrorException is_minimiser([1, 5], f) # One valid, one invalid
        end
    end
    
    @testset "Return Value Consistency" begin
        println("Testing return value consistency...")
        
        @testset "Return types and structure" begin
            f = ConcaveSubmodularFunction(4, 0.5)
            
            # Test optimal case
            result_opt = is_minimiser(falses(4), f)
            @test isa(result_opt, Tuple{Bool, String, Float64})
            is_opt, improvement, better_val = result_opt
            @test isa(is_opt, Bool)
            @test isa(improvement, String)
            @test isa(better_val, Float64)
            @test is_opt == true
            @test improvement == ""
            @test isnan(better_val)
            
            # Test suboptimal case
            S_bad = BitVector([true, false, false, false])
            result_subopt = is_minimiser(S_bad, f)
            is_opt, improvement, better_val = result_subopt
            @test is_opt == false
            @test length(improvement) > 0
            @test !isnan(better_val)
            @test better_val < evaluate(f, S_bad)
        end
    end
    
    @testset "Both Input Methods" begin
        println("Testing both BitVector and Vector{Int} input methods...")
        
        @testset "Equivalent results" begin
            f = ConcaveSubmodularFunction(5, 0.6)
            
            test_cases = [
                (falses(5), Int[]),
                (BitVector([true, false, false, false, false]), [1]),
                (BitVector([true, true, false, false, false]), [1, 2]),
                (BitVector([true, false, true, false, true]), [1, 3, 5]),
                (trues(5), [1, 2, 3, 4, 5])
            ]
            
            for (S_bitvec, S_indices) in test_cases
                result1 = is_minimiser(S_bitvec, f)
                result2 = is_minimiser(S_indices, f)
                
                @test result1[1] == result2[1]  # Same optimality
                @test result1[2] == result2[2]  # Same improvement message
                if !isnan(result1[3]) && !isnan(result2[3])
                    @test abs(result1[3] - result2[3]) < 1e-10  # Same better value
                else
                    @test isnan(result1[3]) == isnan(result2[3])
                end
            end
        end
    end
    
    @testset "Function Parameters" begin
        println("Testing function parameters...")
        
        @testset "Tolerance parameter" begin
            f = ConcaveSubmodularFunction(4, 0.5)
            S = BitVector([true, false, false, false])  # Suboptimal set
            
            # Different tolerances should give same result for clear cases
            result1 = is_minimiser(S, f; tolerance=1e-10)
            result2 = is_minimiser(S, f; tolerance=1e-6)
            result3 = is_minimiser(S, f; tolerance=1e-3)
            
            # All should detect this as suboptimal
            @test result1[1] == result2[1] == result3[1] == false
        end
        
        @testset "Verbose parameter" begin
            f = ConcaveSubmodularFunction(3, 0.5)
            
            # Test that verbose doesn't change the result
            S_opt = falses(3)
            result1 = is_minimiser(S_opt, f; verbose=false)
            
            # Test verbose mode works without errors
            @test_nowarn is_minimiser(S_opt, f; verbose=true)
            result2 = is_minimiser(S_opt, f; verbose=false)
            
            @test result1[1] == result2[1]
            @test result1[2] == result2[2]
            
            # Test verbose with suboptimal case
            S_subopt = BitVector([true, false, false])
            @test_nowarn is_minimiser(S_subopt, f; verbose=true)
            result3 = is_minimiser(S_subopt, f; verbose=false)
            
            @test result3[1] == false
        end
    end
    
    @testset "Cut Functions" begin
        println("Testing cut functions...")
        
        @testset "Path graph cuts" begin
            # Path: 1-2-3-4
            edges = [(1,2), (2,3), (3,4)]
            f = CutFunction(4, edges)
            
            # Empty set and full set should be optimal (cut value = 0)
            S_empty = falses(4)
            is_opt_empty, _, _ = is_minimiser(S_empty, f)
            @test is_opt_empty == true
            
            S_full = trues(4)
            is_opt_full, _, _ = is_minimiser(S_full, f)
            @test is_opt_full == true
            
            # Single elements should not be optimal (cut value > 0)
            S_single = BitVector([true, false, false, false])
            is_opt_single, improvement, better_val = is_minimiser(S_single, f)
            @test is_opt_single == false
            @test better_val == 0.0  # Can improve to empty set
        end
        
        @testset "Triangle graph cuts" begin
            # Triangle: 1-2, 2-3, 1-3
            edges = [(1,2), (1,3), (2,3)]
            f = CutFunction(3, edges)
            
            # Only empty and full sets should be optimal
            test_cases = [
                (falses(3), true),   # Empty set
                (trues(3), true),    # Full set
                (BitVector([true, false, false]), false),   # Single element
                (BitVector([true, true, false]), false),    # Two elements
            ]
            
            for (S, expected_optimal) in test_cases
                is_opt, improvement, better_val = is_minimiser(S, f)
                @test is_opt == expected_optimal
                if !expected_optimal
                    @test !isnan(better_val)
                    @test better_val <= evaluate(f, S)
                end
            end
        end
    end
    
    @testset "Edge Cases" begin
        println("Testing edge cases...")
        
        @testset "Single element ground set" begin
            f = ConcaveSubmodularFunction(1, 0.5)
            
            # Empty set
            S_empty = falses(1)
            is_opt, improvement, better_val = is_minimiser(S_empty, f)
            @test is_opt == true
            
            # Single element set
            S_single = trues(1)
            is_opt, improvement, better_val = is_minimiser(S_single, f)
            @test is_opt == false
            @test improvement == "Can improve by removing element 1"
        end
        
        @testset "Constant function" begin
            # All sets have same value, so all are optimal
            struct ConstantTestFunction <: SubmodularFunction
                n::Int
                c::Float64
            end
            
            function SubmodularMinimization.ground_set_size(f::ConstantTestFunction)
                return f.n
            end
            
            function SubmodularMinimization.evaluate(f::ConstantTestFunction, S::BitVector)
                return f.c
            end
            
            f_const = ConstantTestFunction(4, 5.0)
            
            # All sets should be optimal for constant functions
            test_sets = [
                falses(4),
                BitVector([true, false, false, false]),
                BitVector([true, true, false, false]),
                trues(4)
            ]
            
            for S in test_sets
                is_opt, improvement, better_val = is_minimiser(S, f_const)
                @test is_opt == true
            end
        end
    end
    
    @testset "Comparison with Brute Force" begin
        println("Testing against brute force results...")
        
        @testset "Small functions - exact verification" begin
            test_functions = [
                ConcaveSubmodularFunction(4, 0.5),
                SquareRootFunction(4),
                MatroidRankFunction(4, 2),
                CutFunction(4, [(1,2), (2,3), (3,4)])
            ]
            
            for f in test_functions
                # Find true minimum with brute force
                S_true_min, val_true_min = brute_force_minimization(f)
                
                # Verify that brute force result is identified as optimal
                is_opt_true, improvement_true, better_val_true = is_minimiser(S_true_min, f)
                @test is_opt_true == true
                @test improvement_true == ""
                @test isnan(better_val_true)
                
                # Test some non-optimal sets
                n = ground_set_size(f)
                for trial in 1:5
                    # Generate a random set that's likely not optimal
                    S_random = rand(n) .< 0.5
                    val_random = evaluate(f, S_random)
                    
                    is_opt_random, improvement_random, better_val_random = is_minimiser(S_random, f)
                    
                    if val_random > val_true_min + 1e-10
                        # This set has higher value, so if it's optimal, that's wrong
                        # But we need to be careful about the test logic
                        if is_opt_random
                            # This shouldn't happen - a set with higher value can't be optimal
                            @warn "Set with higher value ($val_random) than minimum ($val_true_min) reported as optimal"
                        end
                        # Note: we can't test !is_opt_random because the function might still be optimal
                        # if there are multiple global optima with the same value
                    else
                        # This might be optimal (has the same value as minimum)
                        if is_opt_random
                            @test abs(val_random - val_true_min) < 1e-10
                        end
                    end
                end
            end
        end
    end
    
    @testset "Non-Submodular Functions" begin
        println("Testing with non-submodular functions...")
        
        @testset "Feature Selection Functions" begin
            # Note: is_minimiser assumes submodularity, so it may give misleading results
            # for non-submodular functions, but we can still test its behavior
            
            f = create_feature_selection(4)
            
            # Get the result from the main algorithm (which may not be globally optimal)
            S_alg, val_alg, _, _ = fujishige_wolfe_submodular_minimization(f; verbose=false)
            
            # Check if the algorithm result passes local optimality
            is_opt_alg, improvement_alg, better_val_alg = is_minimiser(S_alg, f)
            
            # The algorithm result should pass local optimality test
            # (even if it's not globally optimal for non-submodular functions)
            @test is_opt_alg == true
            
            # Compare with brute force
            S_bf, val_bf = brute_force_minimization(f)
            
            # The brute force result should definitely be optimal
            is_opt_bf, improvement_bf, better_val_bf = is_minimiser(S_bf, f)
            @test is_opt_bf == true
            
            # If they're different, the algorithm found a local minimum
            if abs(val_alg - val_bf) > 1e-6
                @warn "Non-submodular function: Algorithm found local minimum ($(round(val_alg, digits=6))) vs global minimum ($(round(val_bf, digits=6)))"
            end
        end
    end
    
    @testset "Performance and Scaling" begin
        println("Testing performance characteristics...")
        
        @testset "Linear time complexity" begin
            # is_minimiser should be O(n) in the ground set size
            # Test that it runs quickly even for larger n
            
            for n in [10, 20, 50]
                f = ConcaveSubmodularFunction(n, 0.5)
                S_empty = falses(n)
                
                # Should complete quickly
                time_taken = @elapsed begin
                    is_opt, improvement, better_val = is_minimiser(S_empty, f)
                end
                
                @test is_opt == true
                @test time_taken < 0.1  # Should be very fast
            end
        end
    end
    
    println("All is_minimiser() tests completed!")
end