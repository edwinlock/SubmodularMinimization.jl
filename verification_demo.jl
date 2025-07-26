"""
Demonstration: Verification vs Finding the Minimizer

This shows that verifying a candidate solution is computationally easier
than finding the minimizer from scratch, for submodular functions.
"""

using SubmodularMinimization
using Random
using BenchmarkTools

"""
    verify_minimizer_submodular(f, S_candidate)

Verify if S_candidate is the global minimizer of submodular function f.
Uses the fact that for submodular functions, local optimality implies global optimality.

For a set S to be optimal, it must satisfy:
f(S ∪ {v}) - f(S) ≥ 0 for all v ∉ S (no beneficial additions)
f(S \ {v}) - f(S) ≤ 0 for all v ∈ S (no beneficial removals)
"""
function verify_minimizer_submodular(f::SubmodularFunction, S_candidate::BitVector)
    n = ground_set_size(f)
    f_S = evaluate(f, S_candidate)
    
    # Check if we can beneficially add any element
    for v in 1:n
        if !S_candidate[v]  # v not in S
            S_plus_v = copy(S_candidate)
            S_plus_v[v] = true
            
            if evaluate(f, S_plus_v) < f_S - 1e-10
                return false, "Can improve by adding element $v"
            end
        end
    end
    
    # Check if we can beneficially remove any element  
    for v in 1:n
        if S_candidate[v]  # v in S
            S_minus_v = copy(S_candidate)
            S_minus_v[v] = false
            
            if evaluate(f, S_minus_v) < f_S - 1e-10
                return false, "Can improve by removing element $v"
            end
        end
    end
    
    return true, "Verified optimal"
end

"""
    verify_minimizer_general(f, S_candidate)

General verification by checking all possible sets (brute force).
This works for any function, not just submodular ones.
"""
function verify_minimizer_general(f::SubmodularFunction, S_candidate::BitVector)
    n = ground_set_size(f)
    if n > 15
        error("General verification only feasible for n ≤ 15")
    end
    
    f_candidate = evaluate(f, S_candidate)
    
    # Check all 2^n possible sets
    for i in 0:(2^n - 1)
        S = BitVector([((i >> (j-1)) & 1) == 1 for j in 1:n])
        f_S = evaluate(f, S)
        
        if f_S < f_candidate - 1e-10
            return false, "Found better solution: $(findall(S)) with value $f_S"
        end
    end
    
    return true, "Verified optimal by exhaustive search"
end

# Demonstration
function main()
    println("=" ^ 70)
    println("VERIFICATION vs FINDING THE MINIMIZER")
    println("=" ^ 70)
    
    Random.seed!(42)
    
    # Test with different problem sizes
    for n in [8, 10, 12]
        println("\n" * "─" ^ 50)
        println("Testing with n = $n")
        println("─" ^ 50)
        
        f = ConcaveSubmodularFunction(n, 0.7)
        
        # 1. FINDING the minimizer (full algorithm)
        println("\n1. FINDING the minimizer:")
        time_finding = @elapsed begin
            S_min, val_min, _, iters = fujishige_wolfe_submodular_minimization(f; verbose=false)
        end
        println("   Time: $(round(time_finding * 1000, digits=2)) ms")
        println("   Iterations: $iters")
        println("   Result: value = $val_min, set = $(findall(S_min))")
        
        # 2. VERIFYING the solution (polynomial time for submodular functions)
        println("\n2. VERIFYING the solution (submodular-specific):")
        time_verify_sub = @elapsed begin
            is_opt_sub, msg_sub = verify_minimizer_submodular(f, S_min)
        end
        println("   Time: $(round(time_verify_sub * 1000, digits=2)) ms")
        println("   Result: $is_opt_sub - $msg_sub")
        
        # 3. VERIFYING by brute force (for comparison, if feasible)
        if n ≤ 12
            println("\n3. VERIFYING by exhaustive search:")
            time_verify_general = @elapsed begin
                is_opt_gen, msg_gen = verify_minimizer_general(f, S_min)
            end
            println("   Time: $(round(time_verify_general * 1000, digits=2)) ms")
            println("   Result: $is_opt_gen - $msg_gen")
            
            # Speedup comparison
            speedup = time_verify_general / time_verify_sub
            println("\n   Submodular verification speedup: $(round(speedup, digits=1))x faster than brute force")
        end
        
        # Overall comparison
        speedup_find_vs_verify = time_finding / time_verify_sub
        println("\n   📊 FINDING vs VERIFYING: $(round(speedup_find_vs_verify, digits=1))x (finding is $(round(speedup_find_vs_verify, digits=1))x slower)")
    end
    
    # Test with a known non-submodular function
    println("\n" * "=" ^ 70)
    println("TESTING WITH NON-SUBMODULAR FUNCTION")
    println("=" ^ 70)
    
    f_nonsubmodular = create_feature_selection(6)
    
    # The algorithm will find a local minimum, not global
    S_local, val_local, _, _ = fujishige_wolfe_submodular_minimization(f_nonsubmodular; verbose=false)
    
    # Verify using submodular-specific method (should pass - it's a local minimum)
    is_opt_sub, msg_sub = verify_minimizer_submodular(f_nonsubmodular, S_local)
    println("Submodular verification: $is_opt_sub - $msg_sub")
    
    # Verify using brute force (should fail - not global minimum)
    S_global, val_global = brute_force_minimization(f_nonsubmodular)
    is_opt_gen, msg_gen = verify_minimizer_general(f_nonsubmodular, S_local)
    println("Global verification: $is_opt_gen - $msg_gen")
    
    println("\nThis shows why checking submodularity first is crucial!")
    println("Local minimum: $val_local, Global minimum: $val_global")
    println("Difference: $(round(val_local - val_global, digits=6))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end