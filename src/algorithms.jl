"""
Fujishige-Wolfe Algorithm Implementation

This module contains the core Wolfe's algorithm and the complete
Fujishige-Wolfe submodular function minimization algorithm.
"""

using LinearAlgebra

"""
Pre-allocated workspace for Wolfe's algorithm.
"""
mutable struct WolfeWorkspace
    n::Int
    max_vertices::Int
    
    # Buffers for linear optimization oracle
    perm_buffer::Vector{Int}
    S_buffer::BitVector
    q_buffer::Vector{Float64}
    
    # Matrix storage for vertices (more efficient than vector-of-vectors)
    S_matrix::Matrix{Float64}  # n × max_vertices matrix to store vertices as columns
    
    # Buffers for affine minimizer
    affine_workspace::AffineWorkspace
    y_buffer::Vector{Float64}
    α_buffer::Vector{Float64}
    
    # Convex combination coefficients
    λ_buffer::Vector{Float64}
    
    # Temporary vectors for algorithm
    x_temp::Vector{Float64}
    x_current::Vector{Float64}  # Current point to avoid allocation
    θ_values::Vector{Float64}
    
    # Index buffers to avoid allocation in findall
    keep_indices_buffer::Vector{Int}
    keep_count::Int
    
    # Enhanced numerical stability tracking
    marginal_stats::Union{Nothing, Tuple{Float64, Float64}}
end

function WolfeWorkspace(n::Int, max_vertices::Int=10*n)
    # Input validation
    n > 0 || throw(ArgumentError("Ground set size n must be positive, got $n"))
    max_vertices > 0 || throw(ArgumentError("max_vertices must be positive, got $max_vertices"))
    max_vertices >= n || throw(ArgumentError("max_vertices ($max_vertices) must be at least n ($n)"))
    
    WolfeWorkspace(
        n,
        max_vertices,
        Vector{Int}(undef, n),
        falses(n),
        Vector{Float64}(undef, n),
        Matrix{Float64}(undef, n, max_vertices),
        AffineWorkspace(n, max_vertices),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, max_vertices),
        Vector{Float64}(undef, max_vertices),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),  # x_current buffer
        Vector{Float64}(undef, max_vertices),
        Vector{Int}(undef, max_vertices),  # keep_indices_buffer
        0,  # keep_count initialized to 0
        nothing  # marginal_stats initialized as nothing
    )
end

"""
    minor_cycles!(workspace::WolfeWorkspace, m::Int)

Perform the minor cycles of Wolfe's algorithm. Updates workspace.x_current in-place.
Returns (success::Bool, new_m::Int) where success indicates normal completion
and new_m is the updated number of vertices.

The minor cycles repeatedly find the minimum norm point in the current affine hull and either:
1. Accept it if all coefficients are non-negative (feasible) - terminates successfully
2. Perform a line search step toward it and remove vertices with zero coefficients - continue

Each iteration of the while loop represents one minor cycle. The function continues
until either the solution is feasible (in the convex hull) or early termination occurs.

# Arguments
- `workspace`: Pre-allocated workspace containing vertex storage and buffers (x_current will be updated)
- `m`: Number of active vertices
"""
function minor_cycles!(workspace::WolfeWorkspace, m::Int)
    while true
        # Compute affine minimizer using matrix storage with enhanced regularization
        affine_minimizer!(workspace.y_buffer, workspace.α_buffer, workspace.affine_workspace, 
                        workspace.S_matrix, m; marginal_stats=workspace.marginal_stats)
        α = workspace.α_buffer[1:m]
        
        # Check if y is in convex hull
        if all(α .>= -NUMERICAL_PRECISION_TOLERANCE)  # Small tolerance for numerical errors
            workspace.x_current .= workspace.y_buffer
            workspace.λ_buffer[1:m] .= α
            return (true, m)
        else
            # Find step size θ using pre-allocated buffer
            num_candidates = 0
            
            # Enhanced step size computation with numerical stability checks
            for i in 1:m
                if α[i] < -NUMERICAL_PRECISION_TOLERANCE && workspace.λ_buffer[i] > α[i] + NUMERICAL_PRECISION_TOLERANCE
                    denominator = workspace.λ_buffer[i] - α[i]
                    if abs(denominator) > 1e-14  # Enhanced numerical stability check
                        θ_candidate = workspace.λ_buffer[i] / denominator
                        if isfinite(θ_candidate) && θ_candidate > 0
                            num_candidates += 1
                            workspace.θ_values[num_candidates] = θ_candidate
                        end
                    end
                end
            end
            
            if num_candidates == 0
                # If no valid θ values, set θ = 1 to move toward y
                θ = 1.0
            else
                θ = minimum(view(workspace.θ_values, 1:num_candidates))
                θ = max(1e-14, min(θ, 1.0))  # Enhanced numerical bounds
            end
            
            # Update x_current and λ
            workspace.x_current .= θ .* workspace.y_buffer .+ (1 - θ) .* workspace.x_current
            for i in 1:m
                workspace.λ_buffer[i] = θ * α[i] + (1 - θ) * workspace.λ_buffer[i]
            end
            
            # Remove vertices with zero coefficients (manual collection to avoid allocation)
            workspace.keep_count = 0
            @inbounds for i in 1:m
                if workspace.λ_buffer[i] > NUMERICAL_PRECISION_TOLERANCE
                    workspace.keep_count += 1
                    workspace.keep_indices_buffer[workspace.keep_count] = i
                end
            end
            
            if workspace.keep_count == 0
                return (false, m)
            end
            
            # Compact the matrix storage by moving kept vertices to the front
            m = workspace.keep_count
            @inbounds for new_idx in 1:m
                old_idx = workspace.keep_indices_buffer[new_idx]
                if new_idx != old_idx
                    workspace.S_matrix[:, new_idx] .= view(workspace.S_matrix, :, old_idx)
                    workspace.λ_buffer[new_idx] = workspace.λ_buffer[old_idx]
                end
            end
            
            # Renormalize
            λ_sum = sum(workspace.λ_buffer[i] for i in 1:m)
            if λ_sum > eps(Float64)
                for i in 1:m
                    workspace.λ_buffer[i] /= λ_sum
                end
            end
            
            if m <= 1
                return (false, m)
            end
        end
    end
end

"""
    wolfe_algorithm!(workspace::WolfeWorkspace, f::SubmodularFunction; 
                     ε::Float64=1e-6, max_iterations::Int=10000, verbose::Bool=false)

Wolfe's algorithm for finding the minimum norm point in the base polytope using 
pre-allocated workspace to minimize memory allocations.
"""
function wolfe_algorithm!(workspace::WolfeWorkspace, f::SubmodularFunction; 
                         ε::Float64=DEFAULT_TOLERANCE, max_iterations::Int=10000, verbose::Bool=false)
    # Input validation
    ground_set_size(f) == workspace.n || throw(ArgumentError("Function ground set size ($(ground_set_size(f))) must match workspace size ($(workspace.n))"))
    ε > 0 || throw(ArgumentError("Tolerance ε must be positive, got $ε"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive, got $max_iterations"))
    
    n = workspace.n
    
    # Initialize with arbitrary vertex using pre-allocated buffers
    fill!(workspace.x_temp, 0.0)
    _, _ = linear_optimization_oracle!(workspace.q_buffer, workspace.perm_buffer, workspace.S_buffer,
                                      workspace.x_temp, f)
    workspace.x_current .= workspace.q_buffer  # Initialize current point
    x = workspace.x_current  # Alias to workspace buffer (no allocation)
    
    # Initialize with matrix storage (more efficient than vector-of-vectors)
    workspace.S_matrix[:, 1] .= x  # Store first vertex as first column
    workspace.λ_buffer[1] = 1.0
    m = 1  # Current number of vertices
    
    iteration = 0
    converged = false
    
    # Stagnation detection variables
    last_gap = Inf
    stagnation_count = 0
    MAX_STAGNATION_ITERATIONS = 50  # Allow 50 iterations without progress
    STAGNATION_TOLERANCE = 1e-11   # If gap improvement is smaller than this, count as stagnation
    
    if verbose
        println("Starting Pre-allocated Wolfe's Algorithm for n=$n")
        println("Initial point norm: $(norm(x))")
    end
    
    while iteration < max_iterations
        iteration += 1
        
        # MAJOR CYCLE - Linear optimization using pre-allocated buffers with enhanced stability
        _, workspace.marginal_stats = linear_optimization_oracle!(workspace.q_buffer, workspace.perm_buffer, workspace.S_buffer,
                                                                 x, f)
        
        # Enhanced termination condition with numerical stability
        x_norm_squared = dot(x, x)  # More numerically stable than norm(x)^2
        x_dot_q = dot(x, workspace.q_buffer)
        gap = x_norm_squared - x_dot_q
        
        # Enhanced tolerance computation with numerical stability
        scale = max(x_norm_squared, abs(x_dot_q), 1.0)
        relative_tolerance = eps(Float64) * scale
        absolute_tolerance = ε^2
        
        # Stagnation detection - check if gap improvement is negligible
        gap_improvement = last_gap - gap
        if abs(gap_improvement) < STAGNATION_TOLERANCE && gap < 1e-10
            stagnation_count += 1
        else
            stagnation_count = 0  # Reset if we made progress
        end
        last_gap = gap
        
        # Enhanced convergence check with better numerical handling
        converged_absolute = gap <= absolute_tolerance
        converged_relative = gap <= relative_tolerance
        converged_stagnation = stagnation_count >= MAX_STAGNATION_ITERATIONS
        
        if converged_absolute || converged_relative
            converged = true
            if verbose
                println("Converged at iteration $iteration with gap $gap")
                println("  Absolute tolerance: $absolute_tolerance") 
                println("  Relative tolerance: $relative_tolerance")
            end
            break
        elseif converged_stagnation
            converged = true
            if verbose
                println("Converged due to stagnation at iteration $iteration with gap $gap")
                println("  No improvement for $stagnation_count iterations (threshold: $MAX_STAGNATION_ITERATIONS)")
                println("  Gap considered effectively zero due to numerical precision limits")
            end
            break
        end
        
        if verbose && (iteration <= 10 || iteration % 100 == 0)
            println("Iteration $iteration: norm = $(round(norm(x), digits=6)), gap = $(round(gap, digits=8)), stagnation = $stagnation_count")
            if gap < 1e-10
                println("  Exact gap: $gap, improvement: $gap_improvement")
            end
        end
        
        # Add new vertex to matrix storage
        m += 1
        if m > workspace.max_vertices
            error("Exceeded maximum vertices ($m > $(workspace.max_vertices))")
        end
        workspace.S_matrix[:, m] .= workspace.q_buffer
        workspace.λ_buffer[m] = 0.0
        
        # MINOR CYCLES - perform line search and vertex cleanup
        success, m = minor_cycles!(workspace, m)
        if !success
            # Early termination in minor cycles
            break
        end
    end
    
    if verbose
        if converged
            println("Algorithm converged in $iteration iterations")
        else
            println("Algorithm did not converge in $max_iterations iterations")
        end
        println("Final point norm: $(norm(x))")
    end
    
    return x, iteration, converged
end

"""
    fujishige_wolfe_submodular_minimization!(workspace::WolfeWorkspace, f::SubmodularFunction; 
                                            ε::Float64=1e-6, max_iterations::Int=10000, verbose::Bool=false)

Complete Fujishige-Wolfe algorithm for submodular function minimization using 
pre-allocated workspace to minimize memory allocations.
"""
function fujishige_wolfe_submodular_minimization!(workspace::WolfeWorkspace, f::SubmodularFunction; 
                                                 ε::Float64=DEFAULT_TOLERANCE, max_iterations::Int=10000, verbose::Bool=false)
    # Input validation
    ground_set_size(f) == workspace.n || throw(ArgumentError("Function ground set size ($(ground_set_size(f))) must match workspace size ($(workspace.n))"))
    ε > 0 || throw(ArgumentError("Tolerance ε must be positive, got $ε"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive, got $max_iterations"))
    
    # Find minimum norm point in base polytope using pre-allocated workspace
    x, iterations, converged = wolfe_algorithm!(workspace, f; ε=ε, max_iterations=max_iterations, verbose=verbose)
    
    # Extract minimizing set using Fujishige theorem with enhanced numerical stability
    # If x* is the minimum norm point in base polytope B(f), 
    # then S* = {i : x*_i < 0} is a minimizing set
    n = workspace.n
    S_min = falses(n)    
    # Adaptive threshold based on solution magnitude for better numerical stability
    solution_scale = maximum(abs, x)
    threshold = max(NUMERICAL_PRECISION_TOLERANCE, solution_scale * 1e-8)
    @inbounds for i in 1:n
        if x[i] < -threshold
            S_min[i] = true
        end
    end

    # Compute minimum value
    min_value = evaluate(f, S_min)
    
    if verbose
        indices = findall(S_min)
        println("Minimizing set indices: $indices")
        println("Minimum value: $min_value")
        println("Set size: $(sum(S_min))")
        println("Solution threshold used: $threshold")
    end
    
    return S_min, min_value, x, iterations
end

"""
    wolfe_algorithm(f::SubmodularFunction; ε::Float64=1e-6, max_iterations::Int=10000, verbose::Bool=false)

Convenience wrapper for wolfe_algorithm! that automatically creates and manages workspace.
For better performance when calling multiple times, create a WolfeWorkspace and use wolfe_algorithm! directly.
"""
function wolfe_algorithm(f::SubmodularFunction; ε::Float64=DEFAULT_TOLERANCE, max_iterations::Int=10000, verbose::Bool=false)
    n = ground_set_size(f)
    # Use a much larger max_vertices to handle worst-case scenarios
    # In the worst case, the algorithm might need up to max_iterations+1 vertices
    workspace = WolfeWorkspace(n, max_iterations+1)
    return wolfe_algorithm!(workspace, f; ε=ε, max_iterations=max_iterations, verbose=verbose)
end

"""
    fujishige_wolfe_submodular_minimization(f::SubmodularFunction; ε::Float64=1e-6, max_iterations::Int=10000, verbose::Bool=false)

Convenience wrapper for fujishige_wolfe_submodular_minimization! that automatically creates and manages workspace.
For better performance when calling multiple times, create a WolfeWorkspace and use fujishige_wolfe_submodular_minimization! directly.
"""
function fujishige_wolfe_submodular_minimization(f::SubmodularFunction; ε::Float64=DEFAULT_TOLERANCE, max_iterations::Int=10000, verbose::Bool=false)
    n = ground_set_size(f)
    # Use a much larger max_vertices to handle worst-case scenarios
    # In the worst case, the algorithm might need up to max_iterations vertices
    workspace = WolfeWorkspace(n, max_iterations)
    return fujishige_wolfe_submodular_minimization!(workspace, f; ε=ε, max_iterations=max_iterations, verbose=verbose)
end