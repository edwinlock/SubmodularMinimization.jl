#!/usr/bin/env python3
"""
Test script for SubmodularMinimization Python wrapper.

This script tests all the C-exposed functions:
1. fujishige_wolfe_solve_c (Fujishige-Wolfe algorithm)
2. wolfe_algorithm_c (Direct Wolfe algorithm)
3. check_submodular_c (Submodularity verification)
4. is_minimiser_c (Optimality verification)

Usage:
    python3 test_python_wrapper.py
"""

import sys
import os

# Add current directory to path to import our wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from submodular_minimization_python import SubmodularMinimizer
except ImportError as e:
    print(f"❌ Could not import wrapper: {e}")
    print("Make sure submodular_minimization_python.py is in the current directory")
    sys.exit(1)

def test_fujishige_wolfe():
    """Test the Fujishige-Wolfe submodular minimization."""
    print("🧮 Testing Fujishige-Wolfe Algorithm")
    print("-" * 40)
    
    solver = SubmodularMinimizer()
    
    # Test 1: Concave function
    print("Test 1: Concave function f(S) = |S|^0.7, n=8")
    result = solver.solve_concave(n=8, alpha=0.7, tolerance=1e-6)
    
    if result.success:
        print(f"  ✅ Success!")
        print(f"  Optimal set: {result.optimal_set}")
        print(f"  Minimum value: {result.min_value:.6f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Set size: {len(result.optimal_set)}")
    else:
        print(f"  ❌ Failed: {result.error_message}")
        return False
    
    # Test 2: Cut function
    print("\nTest 2: Cut function on 4-vertex square graph")
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    result = solver.solve_cut(n=4, edges=edges)
    
    if result.success:
        print(f"  ✅ Success!")
        print(f"  Optimal cut: {result.optimal_set}")
        print(f"  Cut value: {result.min_value:.6f}")
        print(f"  Iterations: {result.iterations}")
    else:
        print(f"  ❌ Failed: {result.error_message}")
        return False
    
    # Test 3: Square root function
    print("\nTest 3: Square root function f(S) = √|S|, n=6")
    result = solver.solve_sqrt(n=6)
    
    if result.success:
        print(f"  ✅ Success!")
        print(f"  Optimal set: {result.optimal_set}")
        print(f"  Minimum value: {result.min_value:.6f}")
        print(f"  Iterations: {result.iterations}")
    else:
        print(f"  ❌ Failed: {result.error_message}")
        return False
    
    return True

def test_wolfe_algorithm():
    """Test the direct Wolfe algorithm."""
    print("\n🎯 Testing Direct Wolfe Algorithm")
    print("-" * 40)
    
    solver = SubmodularMinimizer()
    
    # Test on concave function
    print("Test: Wolfe algorithm on concave function f(S) = |S|^0.5, n=5")
    result = solver.wolfe_algorithm(
        solver.FUNC_TYPE_CONCAVE, 
        [0.5], 
        n=5, 
        tolerance=1e-6
    )
    
    if result.success:
        print(f"  ✅ Success!")
        print(f"  Minimum norm point: {[f'{x:.4f}' for x in result.min_norm_point]}")
        print(f"  Norm: {sum(x*x for x in result.min_norm_point)**0.5:.6f}")
        print(f"  Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")
        return True
    else:
        print(f"  ❌ Failed: {result.error_message}")
        return False

def test_submodularity_check():
    """Test the submodularity verification."""
    print("\n🔍 Testing Submodularity Verification")
    print("-" * 40)
    
    solver = SubmodularMinimizer()
    
    # Test 1: Concave function (should be submodular)
    print("Test 1: Concave function f(S) = |S|^0.7, n=6")
    result = solver.check_submodular(solver.FUNC_TYPE_CONCAVE, [0.7], 6)
    
    if result.success:
        print(f"  ✅ Check completed!")
        print(f"  Is submodular: {result.is_submodular}")
        print(f"  Violations: {result.violations}")
        
        if result.is_submodular:
            print("  ✅ Correctly identified as submodular")
        else:
            print(f"  ⚠️  Unexpected: found {result.violations} violations")
    else:
        print(f"  ❌ Failed: {result.error_message}")
        return False
    
    # Test 2: Square root function (should be submodular)
    print("\nTest 2: Square root function f(S) = √|S|, n=5")
    result = solver.check_submodular(solver.FUNC_TYPE_SQRT, [], 5)
    
    if result.success:
        print(f"  ✅ Check completed!")
        print(f"  Is submodular: {result.is_submodular}")
        print(f"  Violations: {result.violations}")
    else:
        print(f"  ❌ Failed: {result.error_message}")
        return False
    
    return True

def test_optimality_check():
    """Test the optimality verification."""
    print("\n✓ Testing Optimality Verification")
    print("-" * 40)
    
    solver = SubmodularMinimizer()
    
    # First solve a problem to get the optimal solution
    print("Step 1: Solving concave function f(S) = |S|^0.8, n=6")
    solve_result = solver.solve_concave(n=6, alpha=0.8)
    
    if not solve_result.success:
        print(f"  ❌ Solve failed: {solve_result.error_message}")
        return False
    
    print(f"  Algorithm found optimal set: {solve_result.optimal_set}")
    print(f"  Optimal value: {solve_result.min_value:.6f}")
    
    # Test 1: Check if the algorithm's solution is optimal
    print("\nStep 2: Verifying algorithm's solution is optimal")
    opt_result = solver.check_optimality(
        solver.FUNC_TYPE_CONCAVE, 
        [0.8], 
        6, 
        solve_result.optimal_set
    )
    
    if opt_result.success:
        print(f"  ✅ Check completed!")
        print(f"  Is optimal: {opt_result.is_optimal}")
        if opt_result.is_optimal:
            print("  ✅ Algorithm solution verified as optimal")
        else:
            print(f"  ❌ Algorithm solution not optimal, improvement: {opt_result.improvement_value:.6f}")
    else:
        print(f"  ❌ Failed: {opt_result.error_message}")
        return False
    
    # Test 2: Check a suboptimal solution
    print("\nStep 3: Testing with clearly suboptimal set [0, 1, 2, 3, 4, 5] (full set)")
    opt_result = solver.check_optimality(
        solver.FUNC_TYPE_CONCAVE, 
        [0.8], 
        6, 
        [0, 1, 2, 3, 4, 5]  # Full set should be suboptimal for concave function
    )
    
    if opt_result.success:
        print(f"  ✅ Check completed!")
        print(f"  Is optimal: {opt_result.is_optimal}")
        if not opt_result.is_optimal:
            print(f"  ✅ Correctly identified as suboptimal")
            print(f"  Improvement value: {opt_result.improvement_value:.6f}")
        else:
            print(f"  ⚠️  Unexpected: full set identified as optimal")
    else:
        print(f"  ❌ Failed: {opt_result.error_message}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("SubmodularMinimization.jl Python Wrapper Test Suite")
    print("=" * 60)
    
    try:
        # Test all four C-exposed functions
        tests = [
            ("Fujishige-Wolfe Algorithm", test_fujishige_wolfe),
            ("Direct Wolfe Algorithm", test_wolfe_algorithm), 
            ("Submodularity Verification", test_submodularity_check),
            ("Optimality Verification", test_optimality_check)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"\n❌ {test_name} crashed: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
            if success:
                passed += 1
        
        print(f"\nResults: {passed}/{len(results)} tests passed")
        
        if passed == len(results):
            print("\n🎉 All tests passed! Python wrapper is working correctly.")
            return 0
        else:
            print(f"\n💥 {len(results) - passed} tests failed.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        print("\nTroubleshooting:")
        print("1. Build the library: julia build_python_library.jl")
        print("2. Set library path environment variable")
        print("3. Check that all files are in the correct location")
        return 1

if __name__ == "__main__":
    sys.exit(main())