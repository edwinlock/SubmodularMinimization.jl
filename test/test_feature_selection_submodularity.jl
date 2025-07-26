"""
Test whether Feature Selection functions are actually submodular and correctly implemented.

This investigates whether the 24% failure rate is due to:
1. Incorrect implementation
2. Non-submodular functions being generated
3. Genuine algorithmic limitations
"""

using Test
using SubmodularMinimization
using Random
using LinearAlgebra

Random.seed!(42)

@testset "Feature Selection Submodularity Verification" begin
    
    println("Verifying Feature Selection function submodularity...")
    
    @testset "Manual Submodularity Check" begin
        println("Manual submodularity verification...")
        
        # Create a simple, controlled Feature Selection function
        n = 4
        relevance = [2.0, 3.0, 1.0, 4.0]
        redundancy = [0.0 0.2 0.1 0.3;
                     0.2 0.0 0.4 0.2;
                     0.1 0.4 0.0 0.1;
                     0.3 0.2 0.1 0.0]
        α = 0.6
        
        f = FeatureSelectionFunction(n, relevance, redundancy, α)
        
        println("  Testing function: α=$α")
        println("  Relevance: $relevance")
        println("  Redundancy matrix:")
        for i in 1:n
            println("    $(redundancy[i,:])")
        end
        
        # Test submodularity condition: f(S ∪ {v}) - f(S) ≥ f(T ∪ {v}) - f(T) for S ⊆ T, v ∉ T
        violations = 0
        total_tests = 0
        
        # Test all possible combinations
        for s_bits in 0:(2^n - 1)
            for t_bits in s_bits:(2^n - 1)  # T contains S (s_bits ⊆ t_bits)
                # Check if s_bits is actually a subset of t_bits
                if (s_bits & t_bits) != s_bits
                    continue
                end
                
                S = BitVector([((s_bits >> (i-1)) & 1) == 1 for i in 1:n])
                T = BitVector([((t_bits >> (i-1)) & 1) == 1 for i in 1:n])
                
                # Test adding each element v not in T
                for v in 1:n
                    if T[v]  # v is already in T
                        continue
                    end
                    
                    # Create S ∪ {v} and T ∪ {v}
                    S_plus_v = copy(S)
                    S_plus_v[v] = true
                    T_plus_v = copy(T)
                    T_plus_v[v] = true
                    
                    # Calculate marginals
                    marginal_S = evaluate(f, S_plus_v) - evaluate(f, S)
                    marginal_T = evaluate(f, T_plus_v) - evaluate(f, T)
                    
                    total_tests += 1
                    
                    # Check submodularity: marginal_S ≥ marginal_T
                    if marginal_S < marginal_T - 1e-10
                        violations += 1
                        if violations <= 5  # Show first few violations
                            println("  VIOLATION $violations:")
                            println("    S = $(findall(S)), T = $(findall(T)), v = $v")
                            println("    f(S) = $(evaluate(f, S)), f(S∪{v}) = $(evaluate(f, S_plus_v))")
                            println("    f(T) = $(evaluate(f, T)), f(T∪{v}) = $(evaluate(f, T_plus_v))")
                            println("    Marginal on S: $marginal_S")
                            println("    Marginal on T: $marginal_T")
                            println("    Violation: $(marginal_S - marginal_T)")
                        end
                    end
                end
            end
        end
        
        println("  Submodularity test results:")
        println("    Total tests: $total_tests")
        println("    Violations: $violations")
        println("    Success rate: $(round(100 * (total_tests - violations) / total_tests, digits=2))%")
        
        if violations == 0
            println("  ✓ Function is submodular!")
        else
            println("  ✗ Function is NOT submodular!")
        end
        
        @test violations == 0  # Should be submodular
    end
    
    @testset "Implementation Verification" begin
        println("Verifying implementation against manual calculation...")
        
        n = 3
        relevance = [1.0, 2.0, 3.0]
        redundancy = [0.0 0.1 0.2;
                     0.1 0.0 0.3;
                     0.2 0.3 0.0]
        α = 0.7
        
        f = FeatureSelectionFunction(n, relevance, redundancy, α)
        
        # Test specific subsets manually
        test_cases = [
            (BitVector([false, false, false]), "∅"),
            (BitVector([true, false, false]), "{1}"),
            (BitVector([false, true, false]), "{2}"),
            (BitVector([true, true, false]), "{1,2}"),
            (BitVector([true, true, true]), "{1,2,3}")
        ]
        
        for (subset, desc) in test_cases
            # Manual calculation
            if sum(subset) == 0
                expected = 0.0
            else
                relevance_term = sum(relevance[i] for i in 1:n if subset[i])
                redundancy_term = 0.0
                selected = findall(subset)
                for i in selected, j in selected
                    if i < j
                        redundancy_term += redundancy[i,j]
                    end
                end
                expected = -(α * relevance_term - (1 - α) * redundancy_term)
            end
            
            actual = evaluate(f, subset)
            
            println("  $desc: expected=$expected, actual=$actual, match=$(abs(expected - actual) < 1e-10)")
            @test abs(expected - actual) < 1e-10
        end
    end
    
    @testset "Random Function Submodularity" begin
        println("Testing submodularity of randomly generated functions...")
        
        non_submodular_count = 0
        
        for trial in 1:20
            n = rand(3:5)  # Keep small for exhaustive testing
            f = create_feature_selection(n)
            
            # Exhaustive submodularity check
            violations = 0
            total_tests = 0
            
            for s_bits in 0:(2^n - 1)
                for t_bits in s_bits:(2^n - 1)
                    if (s_bits & t_bits) != s_bits
                        continue
                    end
                    
                    S = BitVector([((s_bits >> (i-1)) & 1) == 1 for i in 1:n])
                    T = BitVector([((t_bits >> (i-1)) & 1) == 1 for i in 1:n])
                    
                    for v in 1:n
                        if T[v]
                            continue
                        end
                        
                        S_plus_v = copy(S)
                        S_plus_v[v] = true
                        T_plus_v = copy(T)
                        T_plus_v[v] = true
                        
                        marginal_S = evaluate(f, S_plus_v) - evaluate(f, S)
                        marginal_T = evaluate(f, T_plus_v) - evaluate(f, T)
                        
                        total_tests += 1
                        
                        if marginal_S < marginal_T - 1e-10
                            violations += 1
                        end
                    end
                end
            end
            
            is_submodular = (violations == 0)
            if !is_submodular
                non_submodular_count += 1
                println("  Trial $trial (n=$n): NOT submodular ($violations violations out of $total_tests tests)")
                println("    α=$(f.α)")
                println("    Relevance range: [$(minimum(f.relevance)), $(maximum(f.relevance))]")
                println("    Redundancy range: [$(minimum(f.redundancy)), $(maximum(f.redundancy))]")
            else
                println("  Trial $trial (n=$n): Submodular ✓")
            end
        end
        
        println("  Summary: $(20 - non_submodular_count)/20 functions were submodular")
        println("  Non-submodular rate: $(round(100 * non_submodular_count / 20, digits=1))%")
        
        # If many functions are non-submodular, that's the problem!
        if non_submodular_count > 5
            println("  ⚠️  WARNING: High rate of non-submodular functions detected!")
        end
    end
    
    @testset "Theoretical Analysis" begin
        println("Theoretical analysis of Feature Selection submodularity...")
        
        # The feature selection function is: f(S) = -(α * Σᵢ∈S rᵢ - (1-α) * Σᵢ<j∈S Rᵢⱼ)
        # Let's analyze when this is submodular
        
        println("  Feature Selection function form:")
        println("    f(S) = -(α * relevance_sum - (1-α) * redundancy_sum)")
        println("    where relevance_sum = Σᵢ∈S rᵢ")
        println("    and redundancy_sum = Σᵢ<j∈S Rᵢⱼ")
        println()
        
        # For submodularity, we need: f(S ∪ {v}) - f(S) ≥ f(T ∪ {v}) - f(T) for S ⊆ T
        # Let's derive the marginal gain condition
        
        println("  Marginal gain when adding element v to set S:")
        println("    Δf(S,v) = f(S ∪ {v}) - f(S)")
        println("           = -(α * rᵥ - (1-α) * Σᵢ∈S Rᵢᵥ)")
        println("           = -α * rᵥ + (1-α) * Σᵢ∈S Rᵢᵥ")
        println()
        
        println("  For submodularity, we need:")
        println("    Δf(S,v) ≥ Δf(T,v) for S ⊆ T")
        println("    -α * rᵥ + (1-α) * Σᵢ∈S Rᵢᵥ ≥ -α * rᵥ + (1-α) * Σᵢ∈T Rᵢᵥ")
        println("    (1-α) * Σᵢ∈S Rᵢᵥ ≥ (1-α) * Σᵢ∈T Rᵢᵥ")
        println("    Σᵢ∈S Rᵢᵥ ≥ Σᵢ∈T Rᵢᵥ")
        println()
        
        println("  Since S ⊆ T, we have Σᵢ∈S Rᵢᵥ ≤ Σᵢ∈T Rᵢᵥ (assuming Rᵢᵥ ≥ 0)")
        println("  This means the condition is VIOLATED when redundancy is positive!")
        println("  ⚠️  The Feature Selection function is NOT submodular in general!")
    end
    
    println("Feature Selection submodularity verification completed!")
end