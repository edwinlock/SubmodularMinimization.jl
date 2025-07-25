"""
Tests for Fujishige-Wolfe algorithm implementation
"""

using LinearAlgebra
using Random


@testset "Algorithm Tests" begin
    
    @testset "Wolfe Algorithm Tests" begin
        @testset "Basic Convergence" begin
            # Test with simple concave function
            f = ConcaveSubmodularFunction(4, 0.7)
            
            x, iterations, converged = wolfe_algorithm(f; ε=DEFAULT_TOLERANCE, verbose=false)
            
            @test converged
            @test iterations > 0
            @test iterations < 1000  # Should converge reasonably quickly
            @test length(x) == 4
            @test all(isfinite.(x))
            
            # Check that result is in base polytope (approximately)
            # This is verified by checking x^T q >= ||x||^2 - ε for optimal q
            q = linear_optimization_oracle(x, f)
            gap = norm(x)^2 - dot(x, q)
            @test gap <= COMPARISON_TOLERANCE  # Should satisfy termination condition
        end
        
        @testset "Different Tolerance Levels" begin
            f = ConcaveSubmodularFunction(5, 0.6)
            
            # Test different epsilon values (use reasonable tolerances for floating point)
            tolerances = [LOOSE_TOLERANCE, DEFAULT_TOLERANCE]
            prev_iterations = 0
            
            for ε in tolerances
                x, iterations, converged = wolfe_algorithm(f; ε=ε, verbose=false)
                
                @test converged
                @test iterations >= prev_iterations  # More stringent tolerance should take more iterations
                
                # Verify final gap is within tolerance
                q = linear_optimization_oracle(x, f)
                gap = norm(x)^2 - dot(x, q)
                @test gap <= ε^2 + NUMERICAL_PRECISION_TOLERANCE  # Allow small numerical error
                
                prev_iterations = iterations
            end
        end
        
        @testset "Cut Function Convergence" begin
            # Test with cut function (different structure)
            Random.seed!(101)
            edges = [(1,2), (2,3), (3,4), (1,4), (2,4)]  # Graph with cycle
            f = CutFunction(4, edges)
            
            x, iterations, converged = wolfe_algorithm(f; ε=DEFAULT_TOLERANCE, verbose=false)
            
            @test converged
            @test length(x) == 4
            @test all(isfinite.(x))
            
            # For cut functions, coordinates should be rational numbers (in exact arithmetic)
            # but we'll just check they're reasonable
            @test norm(x) < 10.0  # Should not blow up
        end
        
        @testset "Max Iterations Limit" begin
            f = ConcaveSubmodularFunction(6, 0.8)
            
            # Test with very small max_iterations
            x, iterations, converged = wolfe_algorithm(f; max_iterations=5, verbose=false)
            
            @test iterations <= 5
            # May or may not converge depending on function
            @test length(x) == 6
            @test all(isfinite.(x))
        end
        
        @testset "Deterministic Behavior" begin
            # Algorithm should be deterministic
            f = ConcaveSubmodularFunction(4, 0.5)
            
            x1, iter1, conv1 = wolfe_algorithm(f; ε=DEFAULT_TOLERANCE, verbose=false)
            x2, iter2, conv2 = wolfe_algorithm(f; ε=DEFAULT_TOLERANCE, verbose=false)
            
            @test x1 ≈ x2
            @test iter1 == iter2
            @test conv1 == conv2
        end
    end
    
    @testset "Fujishige-Wolfe Submodular Minimization Tests" begin
        @testset "Concave Function Minimization" begin
            # For f(S) = |S|^α with α < 1, minimum is achieved at empty set
            f = ConcaveSubmodularFunction(5, 0.7)
            
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; verbose=false)
            
            @test length(S_min) == 5
            @test min_val ≈ 0.0 atol=1e-10  # f(∅) = 0
            @test sum(S_min) == 0           # Empty set
        end
        
        @testset "Cut Function Minimization" begin
            # Test with specific cut function where we know the answer
            edges = [(1,2), (2,3), (1,3)]  # Triangle
            f = CutFunction(3, edges)
            
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; verbose=false)
            
            @test min_val == 0  # Minimum cut value is 0 (empty set or full set)
            @test sum(S_min) == 0 || sum(S_min) == 3  # Should be empty or full set
        end
        
        @testset "Verification Against Brute Force" begin
            # Test on small instances where we can verify by brute force
            test_cases = [
                (3, 0.6),  # Small concave function
                (4, 0.8),  # Slightly larger
            ]
            
            for (n, α) in test_cases
                f = ConcaveSubmodularFunction(n, α)
                
                # Fujishige-Wolfe result
                S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=DEFAULT_TOLERANCE, verbose=false)
                
                # Brute force result
                S_bf, val_bf = brute_force_minimization(f)
                
                @test abs(min_val - val_bf) < COMPARISON_TOLERANCE
                # Note: minimizing sets might differ but values should match
            end
        end
        
        @testset "Cut Function Verification" begin
            # Test cut functions against brute force
            Random.seed!(202)
            
            # Small random graphs
            for n in [3, 4]
                edges = Tuple{Int,Int}[]
                for i in 1:n, j in i+1:n
                    if rand() < 0.5
                        push!(edges, (i,j))
                    end
                end
                
                if !isempty(edges)  # Skip if no edges
                    f = CutFunction(n, edges)
                    
                    S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=DEFAULT_TOLERANCE, verbose=false)
                    S_bf, val_bf = brute_force_minimization(f)
                    
                    @test min_val == val_bf  # Cut function values are integers, should match exactly
                end
            end
        end
        
        @testset "Monotonicity Properties" begin
            # Test that smaller epsilon gives more accurate results
            f = ConcaveSubmodularFunction(6, 0.75)
            
            S_min1, min_val1, x1, iter1 = fujishige_wolfe_submodular_minimization(f; ε=LOOSE_TOLERANCE, verbose=false)
            S_min2, min_val2, x2, iter2 = fujishige_wolfe_submodular_minimization(f; ε=DEFAULT_TOLERANCE, verbose=false)
            
            # More precise tolerance should give result at least as good
            @test min_val2 <= min_val1 + 1e-10
            @test iter2 >= iter1  # Should take at least as many iterations
        end
        
        @testset "Fujishige Theorem Correctness" begin
            # Test the set extraction logic
            f = ConcaveSubmodularFunction(4, 0.6)
            
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=DEFAULT_TOLERANCE, verbose=false)
            
            # The extracted set should actually achieve the minimum
            computed_val = evaluate(f, S_min)
            @test abs(computed_val - min_val) < COMPARISON_TOLERANCE
            
            # Test that the minimum norm point x gives correct information
            # Elements in minimizing set should correspond to most negative coordinates
            perm = sortperm(x)
            
            # If S_min[i] = true, then x[i] should be among the smallest coordinates
            for i in 1:4
                if S_min[i]
                    # i should appear early in the sorted permutation
                    pos = findfirst(==(i), perm)
                    @test pos <= sum(S_min)  # Should be among the k smallest
                end
            end
        end
        
        @testset "Algorithm Robustness" begin
            # Test with various functions to ensure robustness
            test_functions = [
                ConcaveSubmodularFunction(3, 0.1),   # Very concave
                ConcaveSubmodularFunction(3, 0.99),  # Nearly linear
                ConcaveSubmodularFunction(7, 0.5),   # Square root
            ]
            
            for f in test_functions
                S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; verbose=false)
                
                @test all(isfinite.(x))
                @test isfinite(min_val)
                @test iterations > 0
                @test length(S_min) == ground_set_size(f)
                
                # Verify the result
                @test abs(evaluate(f, S_min) - min_val) < COMPARISON_TOLERANCE
            end
        end
        
        @testset "Large Instance Performance" begin
            # Test that algorithm scales reasonably
            f = ConcaveSubmodularFunction(15, 0.7)
            
            start_time = time()
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=LOOSE_TOLERANCE, verbose=false)
            elapsed_time = time() - start_time
            
            @test elapsed_time < 10.0  # Should complete in reasonable time
            @test iterations < 500     # Should not take too many iterations
            @test min_val ≈ 0.0 atol=LOOSE_TOLERANCE  # For concave function, minimum is at empty set
        end
    end
    
    @testset "Algorithm State Management" begin
        @testset "Vertex Set Evolution" begin
            # Test that the algorithm properly manages the vertex set S
            f = ConcaveSubmodularFunction(4, 0.8)
            
            # We can't easily test internal state, but we can test that 
            # the algorithm produces valid results under different conditions
            
            # Multiple runs should give consistent results
            results = []
            for _ in 1:5
                S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=DEFAULT_TOLERANCE, verbose=false)
                push!(results, (S_min, min_val, iterations))
            end
            
            # All results should be identical (deterministic algorithm)
            for i in 2:5
                @test results[i][1] == results[1][1]  # Same minimizing set
                @test abs(results[i][2] - results[1][2]) < NUMERICAL_PRECISION_TOLERANCE  # Same minimum value
                @test results[i][3] == results[1][3]  # Same number of iterations
            end
        end
        
        @testset "Numerical Stability" begin
            # Test with functions that might cause numerical issues
            f = ConcaveSubmodularFunction(5, 0.001)  # Very small exponent
            
            S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(f; ε=DEFAULT_TOLERANCE, verbose=false)
            
            @test all(isfinite.(x))
            @test isfinite(min_val)
            @test !any(isnan.(x))
            @test !isnan(min_val)
        end
    end
    
end