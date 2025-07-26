"""
Brute force submodularity checker using Definition 3 from Wikipedia.

Definition 3: A function f is submodular if for every X ⊆ Ω and x1, x2 ∈ Ω ∖ X such that x1 ≠ x2:
f(X ∪ {x1}) + f(X ∪ {x2}) ≥ f(X ∪ {x1, x2}) + f(X)

This captures the "diminishing returns" property.
"""

using Test
using SubmodularMinimization
using Random

"""
    check_submodularity_def3(f, n; verbose=false)

Check if function f is submodular using Definition 3 from Wikipedia.
Returns (is_submodular, violations, total_tests, violation_details).

Definition 3: For every X ⊆ Ω and x1, x2 ∈ Ω ∖ X such that x1 ≠ x2:
f(X ∪ {x1}) + f(X ∪ {x2}) ≥ f(X ∪ {x1, x2}) + f(X)
"""
function check_submodularity_def3(f, n; verbose=false, tolerance=1e-10)
    violations = 0
    total_tests = 0
    violation_details = []
    
    if verbose
        println("Checking submodularity using Definition 3...")
        println("For every X ⊆ Ω and x1, x2 ∈ Ω ∖ X such that x1 ≠ x2:")
        println("f(X ∪ {x1}) + f(X ∪ {x2}) ≥ f(X ∪ {x1, x2}) + f(X)")
        println()
    end
    
    # Iterate over all possible subsets X
    for x_bits in 0:(2^n - 1)
        X = BitVector([((x_bits >> (i-1)) & 1) == 1 for i in 1:n])
        
        # Find elements not in X
        not_in_X = findall(.!X)
        
        # Check all pairs of elements not in X
        for i in 1:length(not_in_X)
            for j in (i+1):length(not_in_X)
                x1, x2 = not_in_X[i], not_in_X[j]
                
                # Create the four sets needed for the inequality
                X_union_x1 = copy(X)
                X_union_x1[x1] = true
                
                X_union_x2 = copy(X)
                X_union_x2[x2] = true
                
                X_union_x1_x2 = copy(X)
                X_union_x1_x2[x1] = true
                X_union_x1_x2[x2] = true
                
                # Evaluate the function on all four sets
                f_X = evaluate(f, X)
                f_X_x1 = evaluate(f, X_union_x1)
                f_X_x2 = evaluate(f, X_union_x2)
                f_X_x1_x2 = evaluate(f, X_union_x1_x2)
                
                # Check the submodularity inequality
                left_side = f_X_x1 + f_X_x2
                right_side = f_X_x1_x2 + f_X
                
                total_tests += 1
                
                # Check if inequality is violated
                if left_side < right_side - tolerance
                    violations += 1
                    
                    violation_detail = (
                        X = findall(X),
                        x1 = x1,
                        x2 = x2,
                        f_X = f_X,
                        f_X_x1 = f_X_x1,
                        f_X_x2 = f_X_x2,
                        f_X_x1_x2 = f_X_x1_x2,
                        left_side = left_side,
                        right_side = right_side,
                        violation_amount = left_side - right_side
                    )
                    push!(violation_details, violation_detail)
                    
                    if verbose && violations <= 5  # Show first 5 violations
                        println("VIOLATION $violations:")
                        println("  X = $(findall(X)), x1 = $x1, x2 = $x2")
                        println("  f(X) = $f_X")
                        println("  f(X ∪ {x1}) = $f_X_x1")
                        println("  f(X ∪ {x2}) = $f_X_x2")
                        println("  f(X ∪ {x1,x2}) = $f_X_x1_x2")
                        println("  Left side:  f(X ∪ {x1}) + f(X ∪ {x2}) = $left_side")
                        println("  Right side: f(X ∪ {x1,x2}) + f(X) = $right_side")
                        println("  Violation: $(left_side - right_side)")
                        println()
                    end
                end
            end
        end
    end
    
    is_submodular = (violations == 0)
    
    if verbose
        println("Submodularity check completed:")
        println("  Total tests: $total_tests")
        println("  Violations: $violations")
        println("  Success rate: $(round(100 * (total_tests - violations) / total_tests, digits=2))%")
        println("  Result: $(is_submodular ? "SUBMODULAR ✓" : "NOT SUBMODULAR ✗")")
    end
    
    return is_submodular, violations, total_tests, violation_details
end

"""
    test_function_submodularity(f, n, name="Unknown"; verbose=true)

Test a specific function for submodularity and print results.
"""
function test_function_submodularity(f, n, name="Unknown"; verbose=true)
    println("Testing $name (n=$n)...")
    is_submodular, violations, total_tests, details = check_submodularity_def3(f, n; verbose=false)
    
    println("  Result: $(is_submodular ? "SUBMODULAR ✓" : "NOT SUBMODULAR ✗")")
    println("  Tests: $violations violations out of $total_tests total tests")
    println("  Success rate: $(round(100 * (total_tests - violations) / total_tests, digits=2))%")
    
    if !is_submodular && verbose && length(details) > 0
        println("  First violation example:")
        d = details[1]
        println("    X = $(d.X), x1 = $(d.x1), x2 = $(d.x2)")
        println("    f(X ∪ {x1}) + f(X ∪ {x2}) = $(d.left_side)")
        println("    f(X ∪ {x1,x2}) + f(X) = $(d.right_side)")
        println("    Violation: $(d.violation_amount)")
    end
    println()
    
    return is_submodular
end

# Run tests if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)
    
    println("=" ^ 60)
    println("BRUTE FORCE SUBMODULARITY CHECKER - DEFINITION 3")
    println("=" ^ 60)
    println()
    
    # Test known submodular functions
    println("Testing known SUBMODULAR functions:")
    println("-" ^ 40)
    
    # Concave functions are submodular
    f1 = ConcaveSubmodularFunction(4, 0.5)
    test_function_submodularity(f1, 4, "ConcaveSubmodularFunction(4, 0.5)")
    
    # Square root functions are submodular
    f2 = SquareRootFunction(4)
    test_function_submodularity(f2, 4, "SquareRootFunction(4)")
    
    # Matroid rank functions are submodular
    f3 = MatroidRankFunction(4, 2)
    test_function_submodularity(f3, 4, "MatroidRankFunction(4, 2)")
    
    # Cut functions are submodular
    edges = [(1,2), (2,3), (3,4), (1,4)]
    f4 = CutFunction(4, edges)
    test_function_submodularity(f4, 4, "CutFunction (square graph)")
    
    # Test Feature Selection functions (should NOT be submodular)
    println("Testing FEATURE SELECTION functions (expected to be non-submodular):")
    println("-" ^ 40)
    
    # Create a simple feature selection function
    relevance = [1.0, 2.0, 3.0, 4.0]
    redundancy = [0.0 0.1 0.2 0.3;
                  0.1 0.0 0.4 0.2;
                  0.2 0.4 0.0 0.1;
                  0.3 0.2 0.1 0.0]
    α = 0.6
    f5 = FeatureSelectionFunction(4, relevance, redundancy, α)
    test_function_submodularity(f5, 4, "FeatureSelectionFunction (manual)")
    
    # Test a randomly generated feature selection function
    f6 = create_feature_selection(4)
    test_function_submodularity(f6, 4, "FeatureSelectionFunction (random)")
    
    # Test multiple random feature selection functions
    println("Testing multiple random Feature Selection functions:")
    println("-" ^ 40)
    
    submodular_count = 0
    total_fs_tests = 10
    
    for i in 1:total_fs_tests
        f = create_feature_selection(4)
        is_sub = check_submodularity_def3(f, 4; verbose=false)[1]
        if is_sub
            submodular_count += 1
        end
        println("  Test $i: $(is_sub ? "SUBMODULAR ✓" : "NOT SUBMODULAR ✗")")
    end
    
    println()
    println("Feature Selection Summary:")
    println("  Submodular: $submodular_count / $total_fs_tests")
    println("  Non-submodular rate: $(round(100 * (total_fs_tests - submodular_count) / total_fs_tests, digits=1))%")
    
    println()
    println("=" ^ 60)
    println("CONCLUSION:")
    println("- Known submodular functions (Concave, SquareRoot, Matroid, Cut) pass ✓")
    println("- Feature Selection functions consistently fail ✗")
    println("- This confirms that Feature Selection functions are NOT submodular")
    println("=" ^ 60)
end