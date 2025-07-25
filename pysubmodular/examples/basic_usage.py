#!/usr/bin/env python3
"""
Basic usage examples for PySubmodular.

This script demonstrates the fundamental functionality of PySubmodular,
including submodular function minimization and direct use of Wolfe's algorithm.
"""

import numpy as np
import pysubmodular as psm


def example_submodular_minimization():
    """Demonstrate basic submodular function minimization."""
    print("=" * 60)
    print("SUBMODULAR FUNCTION MINIMIZATION")
    print("=" * 60)
    
    # Example 1: Concave function f(S) = |S|^0.7
    print("\n1. Concave Function: f(S) = |S|^0.7")
    print("-" * 40)
    
    f1 = psm.ConcaveFunction(n=8, alpha=0.7)
    result1 = psm.minimize_submodular(f1, verbose=True)
    
    print(f"Minimizer set: {result1.selected_indices}")
    print(f"Set size: {result1.set_size}")
    print(f"Minimum value: {result1.min_value:.6f}")
    print(f"Converged in {result1.iterations} iterations")
    print(f"Min norm point: {result1.min_norm_point}")
    
    # Example 2: Square root function f(S) = √|S|
    print("\n2. Square Root Function: f(S) = √|S|")
    print("-" * 40)
    
    f2 = psm.SquareRootFunction(n=6)
    result2 = psm.minimize_submodular(f2)
    
    print(f"Minimizer set: {result2.selected_indices}")
    print(f"Set size: {result2.set_size}")
    print(f"Minimum value: {result2.min_value:.6f}")
    print(f"Converged in {result2.iterations} iterations")
    
    # Example 3: Matroid rank function f(S) = min(|S|, k)
    print("\n3. Matroid Rank Function: f(S) = min(|S|, 3)")
    print("-" * 40)
    
    f3 = psm.MatroidRankFunction(n=10, k=3)
    result3 = psm.minimize_submodular(f3)
    
    print(f"Minimizer set: {result3.selected_indices}")
    print(f"Set size: {result3.set_size}")
    print(f"Minimum value: {result3.min_value:.6f}")
    print(f"Converged in {result3.iterations} iterations")


def example_graph_functions():
    """Demonstrate graph-based submodular functions."""
    print("\n" + "=" * 60)
    print("GRAPH-BASED FUNCTIONS")
    print("=" * 60)
    
    # Example 1: Cut function on a path graph
    print("\n1. Cut Function on Path Graph")
    print("-" * 40)
    
    # Path graph: 0-1-2-3-4
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    f_cut = psm.CutFunction(n=5, edges=edges)
    
    print(f"Graph edges: {edges}")
    
    result_cut = psm.minimize_submodular(f_cut)
    print(f"Minimum cut set: {result_cut.selected_indices}")
    print(f"Cut value: {result_cut.min_value:.6f}")
    
    # Verify by checking a few specific cuts manually
    print("\nManual verification of some cuts:")
    
    # Empty set
    subset = np.array([False, False, False, False, False])
    print(f"Empty set: cut value = {f_cut.evaluate(subset)}")
    
    # Single vertex
    subset = np.array([True, False, False, False, False])
    print(f"{{0}}: cut value = {f_cut.evaluate(subset)}")
    
    # Split in middle
    subset = np.array([True, True, False, False, False])
    print(f"{{0,1}}: cut value = {f_cut.evaluate(subset)}")


def example_wolfe_algorithm():
    """Demonstrate direct use of Wolfe's algorithm."""
    print("\n" + "=" * 60)
    print("WOLFE'S ALGORITHM FOR GENERAL POLYTOPES")
    print("=" * 60)
    
    # Example 1: Standard simplex
    print("\n1. Minimum Norm Point in Standard Simplex")
    print("-" * 40)
    
    def simplex_oracle(c):
        """Linear oracle for standard simplex {x: x ≥ 0, sum(x) = 1}."""
        i_min = np.argmin(c)
        vertex = np.zeros(len(c))
        vertex[i_min] = 1.0
        return vertex
    
    dimension = 5
    result_simplex = psm.wolfe_algorithm(simplex_oracle, dimension)
    
    print(f"Dimension: {dimension}")
    print(f"Min norm point: {result_simplex.min_norm_point}")
    print(f"Norm value: {result_simplex.norm_value:.6f}")
    print(f"Converged in {result_simplex.iterations} iterations")
    print(f"Sum of coordinates: {np.sum(result_simplex.min_norm_point):.6f} (should be 1.0)")
    
    # The minimum norm point in the simplex is (1/n, 1/n, ..., 1/n)
    expected_point = np.ones(dimension) / dimension
    expected_norm = np.linalg.norm(expected_point)
    print(f"Expected point: {expected_point}")
    print(f"Expected norm: {expected_norm:.6f}")
    
    # Example 2: Unit cube
    print("\n2. Minimum Norm Point in Unit Hypercube")
    print("-" * 40)
    
    def cube_oracle(c):
        """Linear oracle for unit hypercube [0,1]^n."""
        # The minimum is achieved at a vertex: either 0 or 1 in each coordinate
        vertex = np.zeros(len(c))
        vertex[c < 0] = 1.0  # Set to 1 where c is negative
        return vertex
    
    dimension = 4
    result_cube = psm.wolfe_algorithm(cube_oracle, dimension)
    
    print(f"Dimension: {dimension}")
    print(f"Min norm point: {result_cube.min_norm_point}")
    print(f"Norm value: {result_cube.norm_value:.6f}")
    print(f"Converged in {result_cube.iterations} iterations")
    
    # The minimum norm point in [0,1]^n is (0.5, 0.5, ..., 0.5)
    expected_point = np.ones(dimension) * 0.5
    expected_norm = np.linalg.norm(expected_point)
    print(f"Expected point: {expected_point}")
    print(f"Expected norm: {expected_norm:.6f}")


def example_facility_location():
    """Demonstrate facility location function."""
    print("\n" + "=" * 60)
    print("FACILITY LOCATION PROBLEM")
    print("=" * 60)
    
    # Create a facility location problem
    n_facilities = 6
    n_customers = 4
    
    # Random weights representing benefit of serving each customer from each facility
    np.random.seed(42)  # For reproducibility
    weights = np.random.rand(n_customers, n_facilities) * 10
    
    print(f"Number of facilities: {n_facilities}")
    print(f"Number of customers: {n_customers}")
    print("Weight matrix (customers × facilities):")
    print(weights)
    
    # Create facility location function
    f_facility = psm.FacilityLocationFunction(n=n_facilities, weights=weights)
    
    # Minimize to find optimal facility placement
    result = psm.minimize_submodular(f_facility)
    
    print(f"\nOptimal facility selection:")
    print(f"Selected facilities: {result.selected_indices}")
    print(f"Number of facilities: {result.set_size}")
    print(f"Total benefit: {result.min_value:.6f}")
    
    # Show which facility serves each customer best
    print(f"\nCustomer assignments:")
    selected_facilities = result.selected_indices
    for customer in range(n_customers):
        customer_weights = weights[customer, selected_facilities]
        best_facility_idx = np.argmax(customer_weights)
        best_facility = selected_facilities[best_facility_idx]
        best_benefit = customer_weights[best_facility_idx]
        print(f"Customer {customer}: served by facility {best_facility} (benefit: {best_benefit:.3f})")


def example_submodularity_check():
    """Demonstrate submodularity checking."""
    print("\n" + "=" * 60)
    print("SUBMODULARITY VERIFICATION")
    print("=" * 60)
    
    # Test various functions for submodularity
    functions = [
        ("Concave(n=6, α=0.8)", psm.ConcaveFunction(6, 0.8)),
        ("SquareRoot(n=5)", psm.SquareRootFunction(5)),
        ("MatroidRank(n=8, k=3)", psm.MatroidRankFunction(8, 3)),
        ("Cut function", psm.CutFunction(4, [(0, 1), (1, 2), (2, 3)])),
    ]
    
    for name, func in functions:
        is_submodular = func.check_submodularity(num_samples=100)
        status = "✓ SUBMODULAR" if is_submodular else "✗ NOT SUBMODULAR"
        print(f"{name:25}: {status}")


def main():
    """Run all examples."""
    print("PySubmodular Examples")
    print("====================")
    
    try:
        example_submodular_minimization()
        example_graph_functions()
        example_wolfe_algorithm()
        example_facility_location()
        example_submodularity_check()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure Julia and SubmodularMinimization.jl are properly installed.")
        raise


if __name__ == "__main__":
    main()