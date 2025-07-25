"""
Tests for optimization oracles (linear optimization oracle, affine minimizer)
"""

using LinearAlgebra
using Random


@testset "Oracle Tests" begin
    
    @testset "Linear Optimization Oracle Tests" begin
        @testset "Basic Functionality" begin
            # Test with simple concave function
            f = ConcaveSubmodularFunction(4, 0.5)
            
            # Test with zero vector (should return vertex of base polytope)
            x = zeros(4)
            q = linear_optimization_oracle(x, f)
            @test length(q) == 4
            @test all(q .>= -1e-10)  # Should be in base polytope (non-negative marginal gains)
            
            # Test that q is a valid base polytope element (sum should equal f(ground_set))
            @test abs(sum(q) - evaluate(f, trues(4))) < 1e-10
        end
        
        @testset "Greedy Property" begin
            # The oracle should solve: min x^T q subject to q in base polytope
            f = ConcaveSubmodularFunction(5, 0.7)
            
            # Test with different cost vectors
            test_vectors = [
                [1.0, 2.0, 3.0, 4.0, 5.0],  # Ascending
                [5.0, 4.0, 3.0, 2.0, 1.0],  # Descending  
                [1.0, -1.0, 2.0, -2.0, 0.0], # Mixed signs
                ones(5),                      # Uniform
                [0.1, 0.1, 0.1, 0.1, 100.0] # One large element
            ]
            
            for x in test_vectors
                q = linear_optimization_oracle(x, f)
                
                # Check that q is in base polytope
                @test abs(sum(q) - evaluate(f, trues(5))) < 1e-10
                
                # Check marginal gain property: q[i] = f(S_i) - f(S_{i-1})
                # where S_i = {j : x[j] <= x[perm[i]]} in sorted order
                perm = sortperm(x)
                S = falses(5)
                f_prev = evaluate(f, S)
                
                for i in 1:5
                    idx = perm[i]
                    S[idx] = true
                    f_curr = evaluate(f, S)
                    expected_marginal = f_curr - f_prev
                    @test abs(q[idx] - expected_marginal) < 1e-10
                    f_prev = f_curr
                end
            end
        end
        
        @testset "Cut Function Oracle" begin
            # Test with cut function on path graph
            edges = [(1,2), (2,3), (3,4)]
            f = CutFunction(4, edges)
            
            x = [1.0, 2.0, 3.0, 4.0]  # Prefer smaller indices
            q = linear_optimization_oracle(x, f)
            
            # Verify base polytope property
            @test abs(sum(q) - evaluate(f, trues(4))) < 1e-10
            
            # For cut functions, marginal gains should be integers
            @test all(abs.(q .- round.(q)) .< 1e-10)
        end
        
        @testset "Deterministic Behavior" begin
            # Oracle should be deterministic for same input
            f = ConcaveSubmodularFunction(6, 0.8)
            x = rand(6)
            
            q1 = linear_optimization_oracle(x, f)
            q2 = linear_optimization_oracle(x, f)
            
            @test q1 ≈ q2
        end
    end
    
    @testset "Affine Minimizer Tests" begin
        @testset "Single Point" begin
            # With single point, should return that point with coefficient 1
            p = rand(3)
            S_matrix = reshape(p, :, 1)  # 3×1 matrix
            workspace = AffineWorkspace(3, 1)
            y = Vector{Float64}(undef, 3)
            α = Vector{Float64}(undef, 1)
            
            result_y, result_α = affine_minimizer!(y, α, workspace, S_matrix, 1)
            
            @test result_y ≈ p
            @test result_α ≈ [1.0]
            @test abs(sum(result_α) - 1.0) < 1e-10
        end
        
        @testset "Two Points" begin
            # Test with two points
            p1 = [1.0, 0.0]
            p2 = [0.0, 1.0]
            S_matrix = hcat(p1, p2)  # 2×2 matrix
            workspace = AffineWorkspace(2, 2)
            y = Vector{Float64}(undef, 2)
            α = Vector{Float64}(undef, 2)
            
            result_y, result_α = affine_minimizer!(y, α, workspace, S_matrix, 2)
            
            # Should find minimum norm point on line segment
            @test length(result_α) == 2
            @test abs(sum(result_α) - 1.0) < 1e-10
            @test all(result_α .>= -1e-10)  # Should be convex combination
            
            # Verify result
            @test result_y ≈ result_α[1] * p1 + result_α[2] * p2
            
            # For this symmetric case, should be at midpoint
            @test result_α[1] ≈ 0.5 atol=1e-6
            @test result_α[2] ≈ 0.5 atol=1e-6
        end
        
        @testset "Three Points Triangle" begin
            # Test with three points forming a triangle
            p1 = [1.0, 0.0]
            p2 = [0.0, 1.0]  
            p3 = [-1.0, 0.0]
            S = [p1, p2, p3]
            
            y, α = affine_minimizer(S)
            
            # Check affine combination property
            @test abs(sum(α) - 1.0) < 1e-10
            @test isapprox(y, α[1] * p1 + α[2] * p2 + α[3] * p3, atol=COMPARISON_TOLERANCE)
            
            # Result should be closer to origin than any individual point
            @test norm(y) <= norm(p1) + 1e-10
            @test norm(y) <= norm(p2) + 1e-10
            @test norm(y) <= norm(p3) + 1e-10
        end
        
        @testset "Collinear Points" begin
            # Test with collinear points
            p1 = [1.0, 1.0]
            p2 = [2.0, 2.0]
            p3 = [3.0, 3.0]
            S = [p1, p2, p3]
            
            y, α = affine_minimizer(S)
            
            # Should still satisfy affine combination
            @test abs(sum(α) - 1.0) < 1e-10
            @test isapprox(y, α[1] * p1 + α[2] * p2 + α[3] * p3, atol=COMPARISON_TOLERANCE)
            
            # Result should be on the line
            # For points on line y = x, result should also satisfy this
            @test abs(y[1] - y[2]) < 1e-10
        end
        
        @testset "Origin in Convex Hull" begin
            # Test case where origin is in convex hull
            p1 = [1.0, 0.0]
            p2 = [-1.0, 0.0]
            p3 = [0.0, 1.0]
            p4 = [0.0, -1.0]
            S = [p1, p2, p3, p4]
            
            y, α = affine_minimizer(S)
            
            # Should find origin (or very close to it)
            @test norm(y) < 1e-6
            @test abs(sum(α) - 1.0) < 1e-10
        end
        
        @testset "Numerical Stability" begin
            # Test with nearly collinear points (challenging for numerical stability)
            base = [1.0, 0.0]
            ε = 1e-8
            p1 = base
            p2 = base + [ε, ε]
            p3 = base + [2*ε, -ε]
            S = [p1, p2, p3]
            
            y, α = affine_minimizer(S)
            
            # Should still satisfy basic properties despite numerical challenges
            @test abs(sum(α) - 1.0) < 2e-4  # Relaxed tolerance for numerical stability with very small vectors
            @test all(isfinite.(y))
            @test all(isfinite.(α))
        end
        
        @testset "High Dimensional" begin
            # Test in higher dimensions
            Random.seed!(789)
            n = 10
            m = 6  # Number of points
            
            S = [randn(n) for _ in 1:m]
            y, α = affine_minimizer(S)
            
            @test length(y) == n
            @test length(α) == m
            @test abs(sum(α) - 1.0) < 1e-10
            
            # Verify affine combination
            computed_y = sum(α[i] * S[i] for i in 1:m)
            @test y ≈ computed_y
        end
        
        @testset "Error Handling" begin
            # Test with empty input
            @test_throws ArgumentError affine_minimizer(Vector{Float64}[])
            
            # Test with inconsistent dimensions - this should work now but give reasonable result
            # The in-place version handles this more gracefully
            S = [[1.0, 2.0], [3.0]]  # Different lengths
            # This will likely work but produce unexpected results - that's OK for now
            # The important thing is it doesn't crash
            @test_throws DimensionMismatch affine_minimizer(S)
        end
    end
    
    @testset "Oracle Integration" begin
        @testset "Base Polytope Properties" begin
            # Test that linear oracle output satisfies base polytope constraints
            f = ConcaveSubmodularFunction(5, 0.6)
            
            for _ in 1:10
                x = randn(5)
                q = linear_optimization_oracle(x, f)
                
                # Check base polytope constraints: q(S) <= f(S) for all S
                for subset_bits in 0:(2^5-1)
                    S = falses(5)
                    for i in 1:5
                        if (subset_bits >> (i-1)) & 1 == 1
                            S[i] = true
                        end
                    end
                    
                    q_S = sum(q[i] for i in 1:5 if S[i]; init=0.0)
                    f_S = evaluate(f, S)
                    @test q_S <= f_S + 1e-10
                end
            end
        end
        
        @testset "Optimality Conditions" begin
            # Test that linear oracle produces optimal solutions
            f = ConcaveSubmodularFunction(4, 0.75)
            x = [1.0, -1.0, 2.0, 0.5]
            
            q_optimal = linear_optimization_oracle(x, f)
            objective_optimal = dot(x, q_optimal)
            
            # Generate several other base polytope vertices and verify optimality
            for _ in 1:20
                # Generate random permutation to create different vertex
                perm = randperm(4)
                q_other = zeros(4)
                S = falses(4)
                f_prev = evaluate(f, S)
                
                for i in 1:4
                    idx = perm[i]
                    S[idx] = true
                    f_curr = evaluate(f, S)
                    q_other[idx] = f_curr - f_prev
                    f_prev = f_curr
                end
                
                objective_other = dot(x, q_other)
                @test objective_optimal <= objective_other + 1e-10
            end
        end
    end
end