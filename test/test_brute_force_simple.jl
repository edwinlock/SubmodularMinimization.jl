"""
Simplified tests for brute force enumeration correctness.

This file tests the core brute force functionality using only built-in functions.
"""

using Test
using SubmodularMinimization
using Random

# Set random seed for reproducibility
Random.seed!(42)

@testset "Brute Force Simple Tests" begin
    
    @testset "Basic Functionality" begin
        println("Testing basic brute force functionality...")
        
        @testset "Small concave function tests" begin
            for n in 2:6
                f = ConcaveSubmodularFunction(n, 0.5)
                S_min, min_val = brute_force_minimization(f)
                
                # For concave functions, minimum should be 0 (empty set)
                @test min_val ≈ 0.0 atol=1e-10
                @test all(x -> !x, S_min)  # Should be empty set
                
                println("  n=$n: min_val = $min_val, |S_min| = $(sum(S_min))")
            end
        end
        
        @testset "Square root function tests" begin
            for n in 2:6
                f = SquareRootFunction(n)
                S_min, min_val = brute_force_minimization(f)
                
                # For square root functions, minimum should be 0 (empty set)
                @test min_val ≈ 0.0 atol=1e-10
                @test all(x -> !x, S_min)  # Should be empty set
                
                println("  n=$n: min_val = $min_val, |S_min| = $(sum(S_min))")
            end
        end
        
        @testset "Matroid rank function tests" begin
            for n in 3:6
                k = div(n, 2)
                f = MatroidRankFunction(n, k)
                S_min, min_val = brute_force_minimization(f)
                
                # For matroid rank functions, minimum should be 0 (empty set)
                @test min_val ≈ 0.0 atol=1e-10
                @test all(x -> !x, S_min)  # Should be empty set
                
                println("  n=$n, k=$k: min_val = $min_val, |S_min| = $(sum(S_min))")
            end
        end
    end
    
    @testset "Enumeration Count Verification" begin
        println("Testing enumeration count...")
        
        @testset "Exhaustive search verification" begin
            # Use verbose mode to check that we evaluate exactly 2^n subsets
            for n in 3:6
                f = ConcaveSubmodularFunction(n, 0.7)
                S_min, min_val, eval_count, values = brute_force_minimization_verbose(f; show_progress=false)
                
                expected_count = 2^n
                @test eval_count == expected_count
                @test length(values) == expected_count
                
                # Check that minimum value appears in the values
                @test min_val ∈ values
                @test minimum(values) == min_val
                
                println("  n=$n: evaluated $eval_count subsets (expected $expected_count)")
            end
        end
    end
    
    @testset "Cut Function Tests" begin
        println("Testing cut functions...")
        
        @testset "Simple path graph" begin
            # Path: 1-2-3-4
            edges = [(1, 2), (2, 3), (3, 4)]
            f = CutFunction(4, edges)
            S_min, min_val = brute_force_minimization(f)
            
            # Minimum cut should be 0 (empty set or full set)
            @test min_val ≈ 0.0 atol=1e-10
            # Should be either empty set or full set
            @test (all(x -> !x, S_min) || all(x -> x, S_min))
            
            println("  Path graph: min_val = $min_val, S_min = $S_min")
        end
        
        @testset "Triangle graph" begin
            # Triangle: 1-2, 2-3, 1-3
            edges = [(1, 2), (1, 3), (2, 3)]
            f = CutFunction(3, edges)
            S_min, min_val = brute_force_minimization(f)
            
            # Minimum cut should be 0 (empty set or full set)
            @test min_val ≈ 0.0 atol=1e-10
            @test (all(x -> !x, S_min) || all(x -> x, S_min))
            
            println("  Triangle graph: min_val = $min_val, S_min = $S_min")
        end
    end
    
    @testset "Edge Cases" begin
        println("Testing edge cases...")
        
        @testset "n=1 case" begin
            f = ConcaveSubmodularFunction(1, 0.5)
            S_min, min_val = brute_force_minimization(f)
            
            @test min_val ≈ 0.0 atol=1e-10
            @test length(S_min) == 1
            @test !S_min[1]  # Should be empty set
        end
        
        @testset "Large n boundary test" begin
            # Test n=20 (maximum allowed)
            f = ConcaveSubmodularFunction(20, 0.5)
            @test_nowarn brute_force_minimization(f)
            
            # Test n=21 (should error)
            f_too_large = ConcaveSubmodularFunction(21, 0.5)
            @test_throws ErrorException brute_force_minimization(f_too_large)
        end
    end
    
    @testset "Detailed Value Analysis" begin
        println("Testing detailed value analysis...")
        
        @testset "Value distribution" begin
            n = 4
            f = ConcaveSubmodularFunction(n, 0.6)
            S_min, min_val, eval_count, values = brute_force_minimization_verbose(f; show_progress=false)
            
            # Basic statistics
            @test length(values) == 16  # 2^4
            @test minimum(values) == min_val
            @test min_val ≈ 0.0 atol=1e-10
            
            # Count frequency of minimum value (should be exactly 1 for concave function)
            min_freq = count(v -> abs(v - min_val) < 1e-10, values)
            @test min_freq == 1  # Only empty set gives minimum
            
            # All values should be non-negative for concave function
            @test all(v -> v >= -1e-10, values)
            
            # Values should be strictly increasing with set size for concave function
            # Check a few specific cases
            empty_val = 0.0^0.6  # Should be 0
            single_val = 1.0^0.6  # Should be 1
            double_val = 2.0^0.6  # Should be 2^0.6
            
            @test empty_val ∈ values
            @test single_val ∈ values
            @test double_val ∈ values
            
            println("  Value statistics: min=$min_val, max=$(maximum(values)), mean=$(round(sum(values)/length(values), digits=4))")
        end
    end
    
    @testset "Function Value Verification" begin
        println("Testing function value verification...")
        
        @testset "Manual subset evaluation" begin
            f = ConcaveSubmodularFunction(3, 0.5)
            
            # Test all 8 possible subsets manually
            expected_values = [
                (falses(3), 0.0),                                    # ∅ -> 0
                (BitVector([true, false, false]), 1.0),             # {1} -> 1^0.5 = 1
                (BitVector([false, true, false]), 1.0),             # {2} -> 1^0.5 = 1  
                (BitVector([false, false, true]), 1.0),             # {3} -> 1^0.5 = 1
                (BitVector([true, true, false]), sqrt(2)),          # {1,2} -> 2^0.5
                (BitVector([true, false, true]), sqrt(2)),          # {1,3} -> 2^0.5
                (BitVector([false, true, true]), sqrt(2)),          # {2,3} -> 2^0.5
                (BitVector([true, true, true]), sqrt(3))            # {1,2,3} -> 3^0.5
            ]
            
            for (subset, expected) in expected_values
                actual = evaluate(f, subset)
                @test actual ≈ expected atol=1e-10
            end
            
            # Verify that brute force finds the minimum correctly
            S_min, min_val = brute_force_minimization(f)
            @test min_val ≈ 0.0 atol=1e-10
            @test S_min == falses(3)
        end
    end
    
    println("All brute force simple tests completed!")
end