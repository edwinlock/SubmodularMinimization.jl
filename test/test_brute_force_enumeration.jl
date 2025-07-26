"""
Tests for brute force enumeration correctness.

This file tests that the brute force implementation correctly enumerates
all possible subsets and finds the true minimum.
"""

using Test
using SubmodularMinimization
using Random

# Import the functions to extend them for custom test structs
import SubmodularMinimization: ground_set_size, evaluate

@testset "Brute Force Enumeration Tests" begin
    
    @testset "Subset Enumeration Correctness" begin
        println("Testing subset enumeration correctness...")
        
        # Test on a simple function where we can verify all evaluations manually
        struct TestCountingFunction <: SubmodularFunction
            n::Int
            evaluations::Vector{BitVector}
        end
        
        
        ground_set_size(f::TestCountingFunction) = f.n
        
        function evaluate(f::TestCountingFunction, S::BitVector)
            push!(f.evaluations, copy(S))
            return sum(S)  # Simple cardinality function
        end
        
        @testset "Small instances (n=1,2,3)" begin
            for n in 1:3
                f = TestCountingFunction(n, BitVector[])
                S_min, min_val = brute_force_minimization(f)
                
                # Check that we evaluated exactly 2^n subsets
                @test length(f.evaluations) == 2^n
                
                # Check that all possible subsets were evaluated
                expected_subsets = Set{BitVector}()
                for i in 0:(2^n - 1)
                    S = falses(n)
                    for j in 1:n
                        if (i >> (j-1)) & 1 == 1
                            S[j] = true
                        end
                    end
                    push!(expected_subsets, S)
                end
                
                actual_subsets = Set(f.evaluations)
                @test actual_subsets == expected_subsets
                
                # For cardinality function, minimum should be empty set with value 0
                @test min_val == 0.0
                @test all(x -> !x, S_min)
            end
        end
        
        @testset "Enumeration order verification (n=4)" begin
            # Test that enumeration follows the expected binary counting order
            f = TestCountingFunction(4, BitVector[])
            brute_force_minimization(f)
            
            # Convert evaluations to integers for easy comparison
            eval_as_ints = [sum(S[j] * 2^(j-1) for j in 1:4) for S in f.evaluations]
            expected_order = collect(0:15)
            
            @test eval_as_ints == expected_order
        end
    end
    
    @testset "Correctness on Known Functions" begin
        println("Testing correctness on functions with known minima...")
        
        @testset "Empty set minimizers" begin
            # Functions where empty set is the unique minimizer
            
            # Concave function f(S) = |S|^α (α < 1)
            for n in 3:6
                for α in [0.3, 0.5, 0.7, 0.9]
                    f = ConcaveSubmodularFunction(n, α)
                    S_min, min_val = brute_force_minimization(f)
                    
                    @test min_val ≈ 0.0 atol=1e-10
                    @test all(x -> !x, S_min)
                end
            end
            
            # Square root function f(S) = √|S|
            for n in 3:6
                f = SquareRootFunction(n)
                S_min, min_val = brute_force_minimization(f)
                
                @test min_val ≈ 0.0 atol=1e-10
                @test all(x -> !x, S_min)
            end
        end
        
        @testset "Non-trivial minimizers" begin
            # Matroid rank function with known structure
            for n in 4:7
                k = div(n, 2)  # k = n/2 (integer division)
                f = MatroidRankFunction(n, k)
                S_min, min_val = brute_force_minimization(f)
                
                # Minimum value should be 0 (empty set)
                @test min_val ≈ 0.0 atol=1e-10
                @test all(x -> !x, S_min)
            end
            
            # Cut function on simple graphs
            @testset "Cut function on path graph" begin
                # Path graph: 1-2-3-4
                edges = [(1, 2), (2, 3), (3, 4)]
                f = CutFunction(4, edges)
                S_min, min_val = brute_force_minimization(f)
                
                # Minimum cut should be 0 (empty set or full set)
                @test min_val ≈ 0.0 atol=1e-10
                @test (all(x -> !x, S_min) || all(x -> x, S_min))
            end
            
            @testset "Cut function on cycle graph" begin
                # Cycle graph: 1-2-3-4-1
                edges = [(1, 2), (2, 3), (3, 4), (1, 4)]
                f = CutFunction(4, edges)
                S_min, min_val = brute_force_minimization(f)
                
                # Minimum cut should be 0 (empty set or full set)
                @test min_val ≈ 0.0 atol=1e-10
                @test (all(x -> !x, S_min) || all(x -> x, S_min))
            end
        end
    end
    
    @testset "Exhaustive Search Verification" begin
        println("Testing exhaustive search verification...")
        
        @testset "Manual verification on small instances" begin
            # Create a custom function where we can manually verify the minimum
            struct ManualTestFunction <: SubmodularFunction
                n::Int
                values::Dict{BitVector, Float64}
            end
            
            ground_set_size(f::ManualTestFunction) = f.n
            evaluate(f::ManualTestFunction, S::BitVector) = f.values[S]
            
            # Test case: n=3 with manually specified values
            values = Dict{BitVector, Float64}()
            for i in 0:7  # 2^3 - 1
                S = falses(3)
                for j in 1:3
                    if (i >> (j-1)) & 1 == 1
                        S[j] = true
                    end
                end
                # Assign arbitrary values, but make {false, true, false} the minimum
                if S == [false, true, false]
                    values[S] = -5.0  # This should be the minimum
                else
                    values[S] = Float64(i)  # Other values are non-negative
                end
            end
            
            f = ManualTestFunction(3, values)
            S_min, min_val = brute_force_minimization(f)
            
            @test min_val ≈ -5.0
            @test S_min == [false, true, false]
        end
        
        @testset "Tie-breaking consistency" begin
            # Test that ties are broken consistently
            struct ConstantFunction <: SubmodularFunction
                n::Int
                value::Float64
            end
            
            ground_set_size(f::ConstantFunction) = f.n
            evaluate(f::ConstantFunction, ::BitVector) = f.value
            
            for n in 2:5
                f = ConstantFunction(n, 1.0)
                S_min, min_val = brute_force_minimization(f)
                
                @test min_val ≈ 1.0
                # Should consistently return the first minimum found (empty set)
                @test all(x -> !x, S_min)
            end
        end
    end
    
    @testset "Edge Cases and Error Handling" begin
        println("Testing edge cases...")
        
        @testset "Size limits" begin
            # Test n=1 (should work)
            f = ConcaveSubmodularFunction(1, 0.5)
            S_min, min_val = brute_force_minimization(f)
            @test min_val ≈ 0.0
            
            # Test n=20 (maximum allowed)
            f_large = ConcaveSubmodularFunction(20, 0.5)
            @test_nowarn brute_force_minimization(f_large)
            
            # Test n=21 (should error)
            f_too_large = ConcaveSubmodularFunction(21, 0.5)
            @test_throws ErrorException brute_force_minimization(f_too_large)
        end
        
        @testset "Numerical precision" begin
            # Test with functions that might have precision issues
            for n in 3:5
                f = ConcaveSubmodularFunction(n, 0.999)  # Very close to 1
                S_min, min_val = brute_force_minimization(f)
                @test min_val ≈ 0.0 atol=1e-10
            end
        end
    end
    
    @testset "Verbose Mode Testing" begin
        println("Testing verbose mode...")
        
        @testset "Progress reporting" begin
            # Test that verbose mode doesn't crash and provides reasonable output
            f = ConcaveSubmodularFunction(5, 0.7)
            
            # Just test that verbose mode runs without crashing
            S_min, min_val, evaluation_count, values_seen = brute_force_minimization_verbose(f; show_progress=false)
            @test min_val ≈ 0.0 atol=1e-10
            @test evaluation_count == 2^5
            @test length(values_seen) == 2^5
            @test all(x -> x >= 0, values_seen)  # All values should be non-negative for concave function
        end
        
        @testset "Statistics collection" begin
            f = ConcaveSubmodularFunction(4, 0.6)
            S_min, min_val, evaluation_count, values_seen = brute_force_minimization_verbose(f; show_progress=false)
            
            # Check basic statistics
            @test evaluation_count == 16  # 2^4
            @test length(values_seen) == 16
            @test minimum(values_seen) == min_val
            @test maximum(values_seen) >= min_val
            @test sum(values_seen)/length(values_seen) > min_val  # Mean should be larger than minimum
            
            # Check that the minimum value appears at least once (for empty set)
            min_count = count(x -> abs(x - min_val) < 1e-10, values_seen)
            @test min_count >= 1
            
            # For concave function, all values should be non-negative and minimum should be 0
            @test min_val ≈ 0.0 atol=1e-10
            @test all(x -> x >= -1e-10, values_seen)  # Allow small numerical errors
        end
    end
end

println("All brute force enumeration tests completed!")