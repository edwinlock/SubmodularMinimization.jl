"""
Optimization Oracles for Submodular Function Minimization

This module contains the linear optimization oracle and affine minimizer
used in the Fujishige-Wolfe algorithm.
"""

using LinearAlgebra

"""
    linear_optimization_oracle(x::Vector{Float64}, f::SubmodularFunction)

Greedy algorithm to solve: min_{q ∈ B_f} x^T q
where B_f is the base polytope of submodular function f.

This is a convenience wrapper that calls the in-place version with temporary allocations.
For better performance with repeated calls, use linear_optimization_oracle! directly.
"""
function linear_optimization_oracle(x::Vector{Float64}, f::SubmodularFunction)
    # Input validation
    n = ground_set_size(f)
    length(x) == n || throw(ArgumentError("Length of x ($(length(x))) must match ground set size ($n)"))
    all(isfinite, x) || throw(ArgumentError("All elements of x must be finite"))
    
    # Create temporary buffers and call in-place version
    q = Vector{Float64}(undef, n)
    perm = Vector{Int}(undef, n)
    S = falses(n)
    
    q, _ = linear_optimization_oracle!(q, perm, S, x, f)
    return q
end

"""
    stabilize_marginal_value(marginal, f_curr, f_prev)

Apply numerical stability checks to marginal values, treating values within machine 
precision as exactly zero to avoid numerical noise. Works with both Float64 and Integer types.
"""
@inline function stabilize_marginal_value(marginal, f_curr, f_prev)
    # Convert to Float64 for computation
    marginal_f = Float64(marginal)
    f_curr_f = Float64(f_curr)
    f_prev_f = Float64(f_prev)
    
    # Numerical stability: treat marginals within machine precision as zero
    threshold = eps(Float64) * max(abs(f_curr_f), abs(f_prev_f), 1.0)
    return abs(marginal_f) < threshold ? 0.0 : marginal_f
end

"""
    track_marginal_statistics!(max_marginal::Ref{Float64}, min_marginal::Ref{Float64}, 
                               marginal::Float64)

Track marginal value statistics for adaptive regularization and numerical analysis.
"""
@inline function track_marginal_statistics!(max_marginal::Ref{Float64}, min_marginal::Ref{Float64}, 
                                           marginal::Float64)
    abs_marginal = abs(marginal)
    max_marginal[] = max(max_marginal[], abs_marginal)
    if abs_marginal > 0
        min_marginal[] = min(min_marginal[], abs_marginal)
    end
end

"""
    compute_adaptive_regularization(max_marginal::Float64, min_marginal::Float64)

Compute adaptive regularization parameter based on the range of marginal values.
Returns a regularization parameter suitable for the current problem's numerical scale.
"""
function compute_adaptive_regularization(max_marginal::Float64, min_marginal::Float64)
    if max_marginal > 0 && min_marginal < Inf
        marginal_ratio = min_marginal / max_marginal
        if marginal_ratio < 1e-10
            return max(1e-14, marginal_ratio * 1e-6)
        end
    end
    return 1e-12  # Default regularization
end

"""
    linear_optimization_oracle!(q::Vector{Float64}, perm::Vector{Int}, S::BitVector,
                                x::Vector{Float64}, f::SubmodularFunction)

Enhanced in-place version of linear_optimization_oracle with improved numerical stability.
- q: output vector (will be overwritten)
- perm: pre-allocated permutation buffer (will be overwritten)  
- S: pre-allocated BitVector for set representation (will be overwritten)
- x: input cost vector
- f: submodular function

Improvements over basic version:
- Stable sorting algorithm for numerical consistency
- Enhanced precision in marginal value computation
- Numerical stability checks for small marginals
- Marginal statistics tracking for adaptive methods
"""
function linear_optimization_oracle!(q::Vector{Float64}, perm::Vector{Int}, S::BitVector,
                                    x::Vector{Float64}, f::SubmodularFunction)
    n = ground_set_size(f)
    
    # Enhanced sorting with numerical stability - use stable MergeSort
    sortperm!(perm, x, alg=MergeSort)
    
    # Incremental evaluation with enhanced precision tracking
    fill!(S, false)
    f_prev = evaluate(f, S)  # f(∅)
    
    # Track marginal statistics for numerical analysis
    max_marginal = Ref(0.0)
    min_marginal = Ref(Inf)
    
    @inbounds for i in 1:n
        idx = perm[i]
        S[idx] = true
        f_curr = evaluate(f, S)
        
        # Enhanced numerical precision for marginal computation
        marginal = f_curr - f_prev
        
        # Apply numerical stability checks
        marginal = stabilize_marginal_value(marginal, f_curr, f_prev)
        
        q[idx] = marginal
        f_prev = f_curr
        
        # Track marginal statistics
        track_marginal_statistics!(max_marginal, min_marginal, marginal)
    end
    
    return q, (max_marginal[], min_marginal[])
end


"""
Pre-allocated workspace for affine minimizer computations.
"""
mutable struct AffineWorkspace
    max_vertices::Int
    B_matrix::Matrix{Float64}
    BtB_matrix::Matrix{Float64}
    BtB_reg::Matrix{Float64}
    e_vector::Vector{Float64}
    α_vector::Vector{Float64}
    y_vector::Vector{Float64}
end

function AffineWorkspace(n::Int, max_vertices::Int)
    AffineWorkspace(
        max_vertices,
        Matrix{Float64}(undef, n, max_vertices),
        Matrix{Float64}(undef, max_vertices, max_vertices),
        Matrix{Float64}(undef, max_vertices, max_vertices),
        Vector{Float64}(undef, max_vertices),
        Vector{Float64}(undef, max_vertices),
        Vector{Float64}(undef, n)
    )
end




"""
    affine_minimizer(S::Vector{Vector{Float64}})

Convenience wrapper for vector-of-vectors input.
Converts to matrix storage internally. For production code, use affine_minimizer! directly.
"""
function affine_minimizer(S::Vector{Vector{Float64}})
    m = length(S)
    m > 0 || throw(ArgumentError("Vector S cannot be empty"))
    
    n = length(S[1])
    
    # Convert to matrix storage with dimension checking
    S_matrix = Matrix{Float64}(undef, n, m)
    for (j, s) in enumerate(S)
        length(s) == n || throw(DimensionMismatch("All vectors must have the same length"))
        S_matrix[:, j] .= s
    end
    
    # Create workspace and call matrix version
    workspace = AffineWorkspace(n, m)
    y = Vector{Float64}(undef, n)
    α = Vector{Float64}(undef, m)
    
    return affine_minimizer!(y, α, workspace, S_matrix, m)
end

"""
    regularize_matrix!(BtB::AbstractMatrix{Float64}; condition_numbers=nothing, 
                      marginal_stats=nothing)

Apply adaptive regularization to the matrix BtB based on its condition number estimate
and optionally the marginal value statistics from the linear optimization oracle.

Uses stronger regularization for more ill-conditioned matrices and adapts based on
the numerical scale of the problem as determined by marginal value ranges.

Returns the condition number estimate for monitoring.
"""
function regularize_matrix!(BtB::AbstractMatrix{Float64}; condition_numbers=nothing, 
                           marginal_stats=nothing)
    m = size(BtB, 1)
    
    # Estimate condition number for adaptive regularization
    max_diag = maximum(abs, diag(BtB))
    min_diag_safe = minimum(abs, diag(BtB)) + eps(Float64)
    condition_estimate = max_diag / min_diag_safe
    
    # Store condition number if monitoring is requested
    if condition_numbers !== nothing
        push!(condition_numbers, condition_estimate)
    end
    
    # Base regularization based on condition number
    if condition_estimate > 1e8
        base_reg = max_diag * 1e-8        # Strong regularization for very ill-conditioned
    elseif condition_estimate > 1e6  
        base_reg = max_diag * 1e-10       # Medium regularization for moderately ill-conditioned
    else
        base_reg = max_diag * 1e-12       # Light regularization for well-conditioned
    end
    
    # Enhanced regularization using marginal statistics if available
    if marginal_stats !== nothing
        max_marginal, min_marginal = marginal_stats
        adaptive_reg = compute_adaptive_regularization(max_marginal, min_marginal)
        # Use the more conservative (larger) of the two regularization parameters
        reg_param = max(base_reg, adaptive_reg * max_diag)
    else
        reg_param = base_reg
    end
    
    # Apply regularization directly to BtB diagonal
    @inbounds for i in 1:m
        BtB[i, i] += reg_param
    end
    
    return condition_estimate
end

"""
    solve_regularized_system!(α::AbstractVector{Float64}, BtB::AbstractMatrix{Float64}, 
                             e::AbstractVector{Float64})

Solve the regularized system BtB * α = e using hierarchical solver strategy.
Uses Cholesky → LU → Pseudoinverse fallback for maximum numerical stability.

The coefficients α are normalized to sum to 1 upon successful solution.
Note: Both α and BtB are modified by this function.
"""
function solve_regularized_system!(α::AbstractVector{Float64}, BtB::AbstractMatrix{Float64}, 
                                  e::AbstractVector{Float64})
    m = size(BtB, 1)
    
    # Hierarchical solver strategy for numerical stability
    try
        # Try Cholesky first (most stable for positive definite matrices)
        # Use non-destructive version to avoid corrupting BtB for fallbacks
        chol_factor = cholesky(Hermitian(BtB, :L))
        α .= e
        ldiv!(chol_factor, α)
    catch
        try
            # Fallback to LU decomposition if Cholesky fails
            # Use non-destructive version
            lu_factor = lu(BtB)
            α .= e
            ldiv!(lu_factor, α)
        catch
            # Last resort: use pseudoinverse
            α .= pinv(BtB) * e
        end
    end
    
    # Normalize coefficients to sum to 1
    sum_α = sum(α)
    if abs(sum_α) > eps(Float64)
        α ./= sum_α
    else
        error("Affine minimization failed: coefficients sum to zero, indicating a degenerate system")
    end
    
    return nothing
end


"""
    affine_minimizer!(y::Vector{Float64}, α::Vector{Float64}, workspace::AffineWorkspace,
                     S::Matrix{Float64}, m::Int; marginal_stats=nothing)

Enhanced in-place version of affine_minimizer using pre-allocated workspace and matrix storage.
- y: pre-allocated output vector for the affine minimizer (will be overwritten)
- α: pre-allocated output vector for coefficients (will be overwritten) 
- workspace: pre-allocated workspace containing all necessary matrices and vectors (will be modified)
- S: matrix where each column is a point (n × m matrix)
- m: number of active columns in S
- marginal_stats: optional tuple (max_marginal, min_marginal) for enhanced regularization

Enhanced features:
- Adaptive regularization based on marginal value statistics
- Better numerical stability through enhanced regularization strategies
"""
function affine_minimizer!(y::Vector{Float64}, α::Vector{Float64}, workspace::AffineWorkspace,
                          S::Matrix{Float64}, m::Int; marginal_stats=nothing)
    # If S is a singleton set, return values are trivial:
    if m == 1
        y .= view(S, :, 1)
        α[1] = 1.0
        return y, view(α, 1:1)
    end
    
    n = size(S, 1)
    
    # Check workspace capacity
    if m > workspace.max_vertices
        error("Number of vertices ($m) exceeds workspace capacity ($(workspace.max_vertices))")
    end
    
    # Use the matrix S directly as B (no copying needed)
    B = view(S, 1:n, 1:m)
    
    try
        # Form BtB using pre-allocated matrix
        BtB = view(workspace.BtB_matrix, 1:m, 1:m)
        mul!(BtB, B', B)
        
        # Use enhanced regularization and solving
        e = view(workspace.e_vector, 1:m)
        fill!(e, 1.0)  
        α_view = view(workspace.α_vector, 1:m)
        
        # Apply enhanced regularization with marginal statistics
        regularize_matrix!(BtB; marginal_stats=marginal_stats)
        solve_regularized_system!(α_view, BtB, e)
        
        # Compute y = B * α
        mul!(y, B, α_view)
        α[1:m] .= α_view
        
        return y, α[1:m]
        
    catch ex
        error("Affine minimization failed in matrix storage version: $(ex)")
    end
end
