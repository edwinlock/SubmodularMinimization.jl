"""
Comprehensive tests for the is_submodular() function.

This test suite verifies that is_submodular() correctly identifies submodular
and non-submodular functions across various cases.
"""

using Test
using SubmodularMinimization
using Random
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

# Define test function types outside of test blocks
struct SupermodularFunction <: SubmodularFunction
    n::Int
end

function SubmodularMinimization.ground_set_size(f::SupermodularFunction)
    return f.n
end

function SubmodularMinimization.evaluate(f::SupermodularFunction, S::BitVector)
    return Float64(sum(S)^2)
end

struct ConstantFunction <: SubmodularFunction
    n::Int
    c::Float64
end

function SubmodularMinimization.ground_set_size(f::ConstantFunction)
    return f.n
end

function SubmodularMinimization.evaluate(f::ConstantFunction, S::BitVector)
    return f.c
end

struct LinearFunction <: SubmodularFunction
    n::Int
    weights::Vector{Float64}
end

function SubmodularMinimization.ground_set_size(f::LinearFunction)
    return f.n
end

function SubmodularMinimization.evaluate(f::LinearFunction, S::BitVector)
    if !any(S)
        return 0.0  # Empty set case
    end
    return sum(f.weights[i] for i in 1:f.n if S[i])
end

@testset "is_submodular() Function Tests" begin
    
    @testset "Known Submodular Functions" begin
        println("Testing known submodular functions...")
        
        @testset "ConcaveSubmodularFunction" begin
            for n in 3:6
                for α in [0.3, 0.5, 0.7, 0.9]
                    f = ConcaveSubmodularFunction(n, α)
                    is_sub, violations, total = is_submodular(f)
                    @test is_sub == true
                    @test violations == 0
                    @test total > 0
                end
            end
        end
        
        @testset "SquareRootFunction" begin
            for n in 3:6
                f = SquareRootFunction(n)
                is_sub, violations, total = is_submodular(f)
                @test is_sub == true
                @test violations == 0
                @test total > 0  # Should have some tests
            end
        end
        
        @testset "MatroidRankFunction" begin
            for n in 3:6
                for k in 1:(n-1)
                    f = MatroidRankFunction(n, k)
                    is_sub, violations, total = is_submodular(f)
                    @test is_sub == true
                    @test violations == 0
                    @test total > 0
                end
            end
        end
        
        @testset "CutFunction" begin
            # Path graph: 1-2-3-4
            edges_path = [(1,2), (2,3), (3,4)]
            f_path = CutFunction(4, edges_path)
            is_sub, violations, total = is_submodular(f_path)
            @test is_sub == true
            @test violations == 0
            
            # Complete graph on 4 vertices
            edges_complete = [(i,j) for i in 1:4 for j in (i+1):4]
            f_complete = CutFunction(4, edges_complete)
            is_sub, violations, total = is_submodular(f_complete)
            @test is_sub == true
            @test violations == 0
            
            # Triangle graph
            edges_triangle = [(1,2), (2,3), (1,3)]
            f_triangle = CutFunction(3, edges_triangle)
            is_sub, violations, total = is_submodular(f_triangle)
            @test is_sub == true
            @test violations == 0
        end
        
        @testset "Random Submodular Functions" begin
            # Test various random submodular functions
            for trial in 1:10
                n = rand(3:5)
                
                # Random cut function
                f_cut = create_random_cut_function(n, 0.4)
                is_sub, violations, total = is_submodular(f_cut)
                @test is_sub == true
                @test violations == 0
                
                # Random facility location
                f_facility = create_random_facility_location(n, rand(2:4))
                is_sub, violations, total = is_submodular(f_facility)
                @test is_sub == true
                @test violations == 0
                
                # Random coverage function
                f_coverage = create_random_coverage_function(n, rand(3:6); coverage_prob=0.3)
                is_sub, violations, total = is_submodular(f_coverage)
                @test is_sub == true
                @test violations == 0
            end
        end
    end
    
    @testset "Known Non-Submodular Functions" begin
        println("Testing known non-submodular functions...")
        
        @testset "FeatureSelectionFunction" begin
            # Test multiple feature selection functions
            for trial in 1:20
                n = rand(3:6)
                f = create_feature_selection(n)
                is_sub, violations, total = is_submodular(f)
                @test is_sub == false
                @test violations > 0
                @test total > 0
                
                # Most feature selection functions should have many violations
                violation_rate = violations / total
                @test violation_rate > 0.1  # At least 10% violations
            end
        end
        
        @testset "Manual Non-Submodular Function" begin
            # Create a simple non-submodular function manually
            # f(S) = |S|^2 (supermodular, so -f is subadditive but not submodular)
            f_super = SupermodularFunction(4)
            is_sub, violations, total = is_submodular(f_super)
            @test is_sub == false
            @test violations > 0
        end
    end
    
    @testset "Edge Cases" begin
        println("Testing edge cases...")
        
        @testset "Small n values" begin
            # n = 1 case
            f1 = ConcaveSubmodularFunction(1, 0.5)
            is_sub, violations, total = is_submodular(f1)
            @test is_sub == true
            @test violations == 0
            @test total == 0  # No pairs to test
            
            # n = 2 case
            f2 = ConcaveSubmodularFunction(2, 0.5)
            is_sub, violations, total = is_submodular(f2)
            @test is_sub == true
            @test violations == 0
            @test total >= 1  # Should have at least 1 test
        end
        
        @testset "Constant function" begin
            # Constant functions are submodular
            f_const = ConstantFunction(4, 5.0)
            is_sub, violations, total = is_submodular(f_const)
            @test is_sub == true
            @test violations == 0
        end
        
        @testset "Linear function" begin
            # Linear functions f(S) = Σᵢ∈S wᵢ are modular (both sub and supermodular)
            weights = [1.0, -2.0, 3.0, -1.5]
            f_linear = LinearFunction(4, weights)
            is_sub, violations, total = is_submodular(f_linear)
            @test is_sub == true
            @test violations == 0
        end
    end
    
    @testset "Function Parameters and Options" begin
        println("Testing function parameters...")
        
        @testset "Tolerance parameter" begin
            f = ConcaveSubmodularFunction(4, 0.5)
            
            # Default tolerance
            is_sub1, violations1, total1 = is_submodular(f)
            
            # Custom tolerance
            is_sub2, violations2, total2 = is_submodular(f; tolerance=1e-8)
            
            # Results should be the same for a clean submodular function
            @test is_sub1 == is_sub2 == true
            @test violations1 == violations2 == 0
            @test total1 == total2
        end
        
        @testset "Verbose parameter" begin
            f = ConcaveSubmodularFunction(3, 0.5)
            
            # Test that verbose doesn't change the result
            is_sub1, violations1, total1 = is_submodular(f; verbose=false)
            
            # Test verbose mode works without errors (don't test output capture)
            @test_nowarn is_submodular(f; verbose=true)
            
            # Results should be the same regardless of verbose setting
            is_sub2, violations2, total2 = is_submodular(f; verbose=false)
            @test is_sub1 == is_sub2
            @test violations1 == violations2  
            @test total1 == total2
        end
    end
    
    @testset "Return Values Validation" begin
        println("Testing return value consistency...")
        
        @testset "Return type consistency" begin
            f = ConcaveSubmodularFunction(4, 0.5)
            result = is_submodular(f)
            
            @test isa(result, Tuple{Bool, Int, Int})
            @test length(result) == 3
            
            is_sub, violations, total = result
            @test isa(is_sub, Bool)
            @test isa(violations, Int)
            @test isa(total, Int)
            @test violations >= 0
            @test total >= 0
            @test violations <= total
        end
        
        @testset "Logical consistency" begin
            # Test multiple functions and verify logical consistency
            functions_and_expected = [
                (ConcaveSubmodularFunction(4, 0.5), true),
                (SquareRootFunction(4), true),
                (MatroidRankFunction(4, 2), true),
                (create_feature_selection(4), false)
            ]
            
            for (f, expected_submodular) in functions_and_expected
                is_sub, violations, total = is_submodular(f)
                
                if expected_submodular
                    @test is_sub == true
                    @test violations == 0
                else
                    @test is_sub == false
                    @test violations > 0
                end
                
                @test total > 0  # Should always have some tests for n >= 3
                @test violations <= total
                
                # Consistency check
                @test (violations == 0) == is_sub
            end
        end
    end
    
    @testset "Performance and Scaling" begin
        println("Testing performance characteristics...")
        
        @testset "Expected number of tests" begin
            # The exact formula is complex, but we can verify basic properties
            for n in 2:6
                f = ConcaveSubmodularFunction(n, 0.5)
                is_sub, violations, total = is_submodular(f)
                
                if n == 1
                    @test total == 0  # No pairs to test
                elseif n == 2
                    @test total >= 1  # Should have at least 1 test
                else
                    @test total > 0   # Should have some tests
                    @test total <= n * (n-1) * 2^n  # Upper bound
                end
            end
        end
        
        @testset "Warning for large n" begin
            # Should warn for n > 15
            f = ConcaveSubmodularFunction(16, 0.5)
            
            # Capture warning
            @test_logs (:warn, r"has O\(n² × 2ⁿ\) complexity") is_submodular(f)
        end
    end
    
    @testset "Comparison with Definition" begin
        println("Testing against mathematical definition...")
        
        @testset "Manual verification for small case" begin
            # For n=3, manually verify a few cases against Definition 3
            f = ConcaveSubmodularFunction(3, 0.5)
            
            # Definition 3: f(X ∪ {x1}) + f(X ∪ {x2}) ≥ f(X ∪ {x1,x2}) + f(X)
            # Test X = ∅, x1 = 1, x2 = 2
            X = falses(3)
            X_x1 = BitVector([true, false, false])
            X_x2 = BitVector([false, true, false]) 
            X_x1_x2 = BitVector([true, true, false])
            
            f_X = evaluate(f, X)
            f_X_x1 = evaluate(f, X_x1)
            f_X_x2 = evaluate(f, X_x2)
            f_X_x1_x2 = evaluate(f, X_x1_x2)
            
            # Check the inequality manually
            left_side = f_X_x1 + f_X_x2
            right_side = f_X_x1_x2 + f_X
            @test left_side >= right_side - 1e-10  # Should hold for submodular function
            
            # This should match what is_submodular finds
            is_sub, violations, total = is_submodular(f)
            @test is_sub == true
            @test violations == 0
        end
    end
    
    @testset "Stress Testing" begin
        println("Stress testing with various functions...")
        
        @testset "Many random submodular functions" begin
            successes = 0
            total_tests = 50
            
            for i in 1:total_tests
                n = rand(3:5)
                func_type = rand(1:4)
                
                if func_type == 1
                    f = ConcaveSubmodularFunction(n, 0.1 + rand() * 0.8)
                elseif func_type == 2
                    f = SquareRootFunction(n)
                elseif func_type == 3
                    f = MatroidRankFunction(n, rand(1:(n-1)))
                else
                    f = create_random_cut_function(n, 0.2 + rand() * 0.4)
                end
                
                is_sub, violations, total_sub_tests = is_submodular(f)
                if is_sub
                    successes += 1
                    @test violations == 0
                else
                    # This should not happen for known submodular functions
                    @warn "Unexpected non-submodular result for function type $func_type, n=$n"
                end
            end
            
            # All should be submodular
            @test successes == total_tests
        end
        
        @testset "Many random feature selection functions" begin
            failures = 0
            total_tests = 50
            
            for i in 1:total_tests
                n = rand(3:5)
                f = create_feature_selection(n)
                
                is_sub, violations, total_sub_tests = is_submodular(f)
                if !is_sub
                    failures += 1
                    @test violations > 0
                end
            end
            
            # Most (ideally all) should be non-submodular
            failure_rate = failures / total_tests
            @test failure_rate > 0.8  # At least 80% should fail submodularity
        end
    end
    
    println("All is_submodular() tests completed!")
end