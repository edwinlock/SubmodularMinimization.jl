"""
Tests for edge cases and error handling
"""

using LinearAlgebra
using Random

@testset "Edge Cases and Error Handling" begin
    
    @testset "Invalid Function Parameters" begin
        @testset "ConcaveSubmodularFunction Invalid α" begin
            # Test boundary values for α
            @test_throws ArgumentError ConcaveSubmodularFunction(5, 0.0)    # α = 0
            @test_throws ArgumentError ConcaveSubmodularFunction(5, 1.0)    # α = 1
            @test_throws ArgumentError ConcaveSubmodularFunction(5, -0.1)   # α < 0
            @test_throws ArgumentError ConcaveSubmodularFunction(5, 1.1)    # α > 1
            @test_throws ArgumentError ConcaveSubmodularFunction(5, -Inf)   # α = -∞
            @test_throws ArgumentError ConcaveSubmodularFunction(5, Inf)    # α = ∞
            @test_throws ArgumentError ConcaveSubmodularFunction(5, NaN)    # α = NaN
            
            # Test valid boundary cases
            f1 = ConcaveSubmodularFunction(5, 0.001)  # Very small α
            @test f1.α == 0.001
            
            f2 = ConcaveSubmodularFunction(5, 0.999)  # Very large α  
            @test f2.α == 0.999
        end
        
        @testset "Invalid Ground Set Size" begin
            # Test invalid ground set sizes
            @test_throws Exception ConcaveSubmodularFunction(0, 0.5)   # n = 0
            @test_throws Exception ConcaveSubmodularFunction(-1, 0.5)  # n < 0
            
            # Very large n should work (though may be slow)
            f = ConcaveSubmodularFunction(1000, 0.5)
            @test ground_set_size(f) == 1000
        end
        
        @testset "CutFunction Invalid Edges" begin
            # Test invalid edge specifications
            n = 4
            
            # Edges with invalid vertex indices  
            @test_throws BoundsError evaluate(CutFunction(n, [(0, 1)]), trues(n))  # Vertex 0
            @test_throws BoundsError evaluate(CutFunction(n, [(1, 5)]), trues(n))  # Vertex > n
            @test_throws BoundsError evaluate(CutFunction(n, [(-1, 2)]), trues(n)) # Negative vertex
            
            # Self-loops are not allowed in cut functions
            @test_throws ArgumentError CutFunction(3, [(1, 1), (2, 3)])  # Self-loop (1,1) should throw error
        end
    end
    
    @testset "Degenerate Cases" begin
        @testset "Single Vertex Functions" begin
            # Ground set with single element
            f = ConcaveSubmodularFunction(1, 0.6)
            
            # Test evaluation
            @test evaluate(f, falses(1)) == 0.0
            @test evaluate(f, trues(1)) == 1.0
            
            # Test algorithm
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; verbose=false)
            @test min_val == 0.0
            @test S_min == falses(1)
            @test length(x) == 1
            
            # Test enhanced version (same as standard since optimized was removed)  
            S_enh, val_enh, x_enh, iter_enh = fujishige_wolfe_submodular_minimization(f; verbose=false)
            @test val_enh == 0.0
            @test S_enh == falses(1)
        end
        
        @testset "Empty Edge Set" begin
            # Cut function with no edges
            f = CutFunction(5, Tuple{Int,Int}[])
            
            # All sets should have cut value 0
            @test evaluate(f, falses(5)) == 0
            @test evaluate(f, trues(5)) == 0
            @test evaluate(f, BitVector([1,0,1,0,1])) == 0
            
            # Algorithm should work
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; verbose=false)
            @test min_val == 0
        end
        
        @testset "Complete Graph" begin
            # Complete graph where every subset gives same cut value
            n = 4
            edges = [(i,j) for i in 1:n for j in i+1:n]  # All possible edges
            f = CutFunction(n, edges)
            
            # Empty set and full set should have cut value 0
            @test evaluate(f, falses(n)) == 0
            @test evaluate(f, trues(n)) == 0
            
            # Algorithm should find minimum
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; verbose=false)
            @test min_val == 0
        end
    end
    
    @testset "Numerical Edge Cases" begin
        @testset "Very Small Tolerance" begin
            f = ConcaveSubmodularFunction(5, 0.7)
            
            # Test with extremely small tolerance
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=1e-15, verbose=false)
            @test isfinite(min_val)
            @test all(isfinite.(x))
            @test !isnan(min_val)
            @test !any(isnan.(x))
            
            # Should still converge (though may take many iterations)
            @test iterations > 0
        end
        
        @testset "Very Large Tolerance" begin
            f = ConcaveSubmodularFunction(6, 0.8)
            
            # Test with very large tolerance (should converge quickly)
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=1e-1, verbose=false)
            @test isfinite(min_val)
            @test all(isfinite.(x))
            @test iterations >= 1
        end
        
        @testset "Extreme Function Values" begin
            # Test with function that produces very small values
            f = ConcaveSubmodularFunction(4, 0.01)  # f(S) = |S|^0.01
            
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; verbose=false)
            @test min_val ≈ 0.0 atol=1e-10
            @test all(isfinite.(x))
            
            # Test with function that produces larger values  
            f_large = ConcaveSubmodularFunction(3, 0.99)  # f(S) = |S|^0.99 ≈ |S|
            
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f_large; verbose=false)
            @test min_val ≈ 0.0 atol=1e-10
            @test all(isfinite.(x))
        end
    end
    
    @testset "Algorithm Limits" begin
        @testset "Maximum Iterations" begin
            f = ConcaveSubmodularFunction(6, 0.75)
            
            # Test with limited maximum iterations (must be >= n for workspace)
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; max_iterations=10, verbose=false)
            @test iterations <= 10
            @test all(isfinite.(x))
            @test isfinite(min_val)
            
            # Test with more reasonable limit
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; max_iterations=100, verbose=false)
            @test iterations <= 100
            @test all(isfinite.(x))
        end
        
        @testset "Convergence Issues" begin
            # Test function that might be challenging for convergence
            f = ConcaveSubmodularFunction(8, 0.999)  # Nearly linear
            
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=1e-6, max_iterations=1000, verbose=false)
            
            # Should still produce valid results even if challenging
            @test all(isfinite.(x))
            @test isfinite(min_val)
            @test !any(isnan.(x))
            @test !isnan(min_val)
        end
    end
    
    @testset "Memory and Performance Edge Cases" begin
        @testset "Large Ground Set" begin
            # Test with larger ground set (but still reasonable for testing)
            f = ConcaveSubmodularFunction(20, 0.6)
            
            # Should complete without memory issues
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=1e-4, verbose=false)
            @test length(S_min) == 20
            @test length(x) == 20
            @test min_val ≈ 0.0 atol=1e-3
        end
        
        @testset "Dense Cut Function" begin
            # Test cut function with many edges
            n = 12
            edges = [(i,j) for i in 1:n for j in i+1:n if rand() < 0.7]  # Dense random graph
            f = CutFunction(n, edges)
            
            # Should handle dense graphs without issues
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=1e-4, verbose=false)
            @test isfinite(min_val)
            @test all(isfinite.(x))
            @test min_val >= 0  # Cut values are non-negative
        end
    end
    
    @testset "Input Validation" begin
        @testset "BitVector Size Mismatch" begin
            f = ConcaveSubmodularFunction(5, 0.6)
            
            # Test with wrong-sized BitVector
            @test_throws ArgumentError evaluate(f, falses(4))  # Too small
            @test_throws ArgumentError evaluate(f, falses(6))  # Too large
            
            # Correct size should work
            @test evaluate(f, falses(5)) == 0.0
        end
        
        @testset "Oracle Input Validation" begin
            f = ConcaveSubmodularFunction(4, 0.7)
            
            # Test linear oracle with wrong-sized vector
            @test_throws ArgumentError linear_optimization_oracle(zeros(3), f)  # Too small
            @test_throws ArgumentError linear_optimization_oracle(zeros(5), f)  # Too large
            
            # Correct size should work
            q = linear_optimization_oracle(zeros(4), f)
            @test length(q) == 4
        end
        
        @testset "Affine Minimizer Edge Cases" begin
            # Test affine minimizer with problematic inputs
            
            # Empty vector of points
            @test_throws ArgumentError affine_minimizer(Vector{Float64}[])
            
            # Points with inconsistent dimensions - now throws proper error
            S_inconsistent = [[1.0, 2.0], [3.0]]  # Different lengths
            @test_throws DimensionMismatch affine_minimizer(S_inconsistent)
            
            # Single point (should work)
            S_single = [[1.0, -2.0, 3.0]]
            y, α = affine_minimizer(S_single)
            @test y ≈ S_single[1]
            @test α ≈ [1.0]
        end
    end
    
    @testset "Floating Point Edge Cases" begin
        @testset "Infinity and NaN Handling" begin
            # These should be caught by assertions or produce reasonable behavior
            
            # Test with very large but finite values
            f = ConcaveSubmodularFunction(3, 0.5)
            x_large = [1e10, -1e10, 0.0]
            
            # Should not crash
            q = linear_optimization_oracle(x_large, f)
            @test all(isfinite.(q))
            @test !any(isnan.(q))
        end
        
        @testset "Precision Loss" begin
            # Test cases that might lead to precision loss
            f = ConcaveSubmodularFunction(5, 0.1)  # Very small exponent
            
            # Should handle small function values gracefully
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=1e-12, verbose=false)
            @test all(isfinite.(x))
            @test isfinite(min_val)
            @test min_val >= 0  # Should be non-negative
        end
    end
    
    @testset "Brute Force Edge Cases" begin
        @testset "Size Limit Enforcement" begin
            # Brute force should reject large instances
            f_large = ConcaveSubmodularFunction(25, 0.6)
            @test_throws ErrorException brute_force_minimization(f_large)
            
            # Should work for size limit
            f_small = ConcaveSubmodularFunction(4, 0.6)
            S_min, min_val = brute_force_minimization(f_small)
            @test length(S_min) == 4
            @test isfinite(min_val)
        end
        
        @testset "Brute Force Correctness on Edge Cases" begin
            # Test brute force on degenerate cases
            
            # Single element
            f1 = ConcaveSubmodularFunction(1, 0.7)
            S_min, min_val = brute_force_minimization(f1)
            @test min_val == 0.0  # min(f(∅), f({1})) = min(0, 1) = 0
            @test S_min == falses(1)
            
            # Two elements
            f2 = ConcaveSubmodularFunction(2, 0.6)
            S_min, min_val = brute_force_minimization(f2)
            @test min_val == 0.0  # Empty set is minimum
            @test S_min == falses(2)
        end
    end
    
    @testset "Thread Safety Edge Cases" begin
        @testset "Concurrent Access" begin
            # Test that functions can be evaluated concurrently without issues
            f = ConcaveSubmodularFunction(8, 0.7)
            test_sets = [BitVector(rand(Bool, 8)) for _ in 1:10]
            
            # Sequential evaluation
            results_seq = [evaluate(f, S) for S in test_sets]
            
            # Should not crash with concurrent access (basic test)
            # Note: Full thread safety testing would require more sophisticated setup
            if Threads.nthreads() > 1
                results_threaded = Vector{Float64}(undef, length(test_sets))
                Threads.@threads for i in 1:length(test_sets)
                    results_threaded[i] = evaluate(f, test_sets[i])
                end
                @test results_threaded == results_seq
            end
        end
    end
end