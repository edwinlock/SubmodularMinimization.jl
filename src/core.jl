"""
Core Submodular Function Minimization Types and Interface

This module defines the abstract types and core interface for submodular functions.
"""

"""
    SubmodularFunction

Abstract type for submodular functions. Concrete implementations should define:
- evaluate(f, S): evaluate f on subset S (where S is a BitVector)
- ground_set_size(f): return the size of the ground set
"""
abstract type SubmodularFunction end

"""
    evaluate(f::SubmodularFunction, S::BitVector)

Evaluate the submodular function f on subset S represented as a BitVector.
"""
function evaluate end

"""
    ground_set_size(f::SubmodularFunction)

Return the size of the ground set for the submodular function.
"""
function ground_set_size end