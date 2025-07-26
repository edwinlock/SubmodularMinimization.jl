"""
Python Interface for SubmodularMinimization.jl

This module provides C-compatible wrapper functions for use with PackageCompiler.jl
to create a dynamic library that can be called from Python via ctypes.
"""

using SubmodularMinimization

# Function type constants (exported as enum-like constants)
const FUNC_TYPE_CONCAVE = Cint(1)
const FUNC_TYPE_CUT = Cint(2)
const FUNC_TYPE_SQRT = Cint(3)
const FUNC_TYPE_MATROID = Cint(4)
const FUNC_TYPE_BIPARTITE_MATCHING = Cint(5)
const FUNC_TYPE_FACILITY_LOCATION = Cint(6)
const FUNC_TYPE_CALLBACK = Cint(7)

# Error codes
const SUCCESS = Cint(0)
const ERROR_INVALID_FUNCTION_TYPE = Cint(-1)
const ERROR_INVALID_PARAMETERS = Cint(-2)
const ERROR_CONVERGENCE_FAILED = Cint(-3)
const ERROR_MEMORY_ALLOCATION = Cint(-4)

# Global storage for Python callback function and memoization cache
const PYTHON_CALLBACK = Ref{Ptr{Cvoid}}(C_NULL)
const MEMOIZATION_CACHE = Dict{Vector{Int}, Float64}()
const CACHE_HITS = Ref{Int}(0)
const CACHE_MISSES = Ref{Int}(0)

"""
CallbackSubmodularFunction

A submodular function that calls back to Python for evaluation.
"""
struct CallbackSubmodularFunction <: SubmodularFunction
    n::Int
    callback_ptr::Ptr{Cvoid}
end

ground_set_size(f::CallbackSubmodularFunction) = f.n

function evaluate(f::CallbackSubmodularFunction, S::BitVector)
    # Convert BitVector to array of indices (0-indexed for Python)
    indices = [i - 1 for i in findall(S)]  # Convert to 0-indexed
    
    # Check memoization cache first
    if haskey(MEMOIZATION_CACHE, indices)
        CACHE_HITS[] += 1
        return MEMOIZATION_CACHE[indices]
    end
    
    # Cache miss - call Python function
    CACHE_MISSES[] += 1
    n_indices = length(indices)
    
    # Create C arrays for the callback
    indices_array = Cint.(indices)
    
    # Call Python function through C interface
    # The callback signature should be: double callback(int* indices, int n_indices)
    if f.callback_ptr != C_NULL
        result = ccall(f.callback_ptr, Cdouble, (Ptr{Cint}, Cint), indices_array, Cint(n_indices))
        result_float = Float64(result)
        
        # Store in cache for future use
        MEMOIZATION_CACHE[copy(indices)] = result_float
        
        return result_float
    else
        error("Python callback not set")
    end
end

"""
Create a submodular function from type and parameters.
Returns the function object or nothing if invalid.
"""
function create_function_from_params(func_type::Cint, params::Vector{Float64}, n::Cint)
    try
        if func_type == FUNC_TYPE_CONCAVE
            # params[1] = alpha parameter
            length(params) >= 1 || return nothing
            return ConcaveSubmodularFunction(Int(n), params[1])
        elseif func_type == FUNC_TYPE_CUT
            # params = flat array of edges: [u1, v1, u2, v2, ...]
            length(params) % 2 == 0 || return nothing
            edges = [(Int(params[i]), Int(params[i+1])) for i in 1:2:length(params)]
            return CutFunction(Int(n), edges)
        elseif func_type == FUNC_TYPE_SQRT
            # No parameters needed
            return SquareRootFunction(Int(n))
        elseif func_type == FUNC_TYPE_MATROID
            # params[1] = k (rank limit)
            length(params) >= 1 || return nothing
            return MatroidRankFunction(Int(n), Int(params[1]))
        elseif func_type == FUNC_TYPE_BIPARTITE_MATCHING
            # params[1] = n1, params[2] = n2, params[3] = density
            length(params) >= 3 || return nothing
            return create_bipartite_matching(Int(params[1]), Int(params[2]), params[3])
        elseif func_type == FUNC_TYPE_FACILITY_LOCATION
            # params[1] = num_facilities, params[2] = num_clients
            # Remaining params are weights matrix (flattened)
            length(params) >= 2 || return nothing
            num_facilities = Int(params[1])
            num_clients = Int(params[2])
            if length(params) >= 2 + num_facilities * num_clients
                weights = reshape(params[3:end], num_facilities, num_clients)
                return FacilityLocationFunction(num_clients, weights)
            else
                # Use random weights if not provided
                return create_random_facility_location(num_facilities, num_clients)
            end
        elseif func_type == FUNC_TYPE_CALLBACK
            # params[1] = n (ground set size)
            # Use the globally registered callback
            length(params) >= 1 || return nothing
            n = Int(params[1])
            return CallbackSubmodularFunction(n, PYTHON_CALLBACK[])
        else
            return nothing
        end
    catch e
        @warn "Error creating function: $e"
        return nothing
    end
end

"""
    fujishige_wolfe_solve_c(func_type, params, n_params, n, tolerance, max_iterations, 
                           result_set, result_value, result_iterations)

Fujishige-Wolfe submodular minimization solver callable from C/Python.

# Arguments
- `func_type::Cint`: Function type constant
- `params::Ptr{Cdouble}`: Pointer to parameter array
- `n_params::Cint`: Number of parameters
- `n::Cint`: Ground set size
- `tolerance::Cdouble`: Convergence tolerance
- `max_iterations::Cint`: Maximum iterations
- `result_set::Ptr{Ptr{Cint}}`: Output pointer to array for optimal set indices (allocated by Julia)
- `result_value::Ptr{Cdouble}`: Output pointer for minimum value
- `result_iterations::Ptr{Cint}`: Output pointer for iteration count

# Returns
- `Cint`: Number of elements in optimal set (>= 0), or error code (< 0)
"""
Base.@ccallable function fujishige_wolfe_solve_c(
    func_type::Cint,
    params::Ptr{Cdouble},
    n_params::Cint,
    n::Cint,
    tolerance::Cdouble,
    max_iterations::Cint,
    result_set::Ptr{Ptr{Cint}},     # Pointer to pointer (Julia allocates)
    result_value::Ptr{Cdouble},
    result_iterations::Ptr{Cint}
)::Cint
    try
        # Extract parameters from C array
        param_array = unsafe_wrap(Array, params, Int(n_params))
        
        # Create function
        f = create_function_from_params(func_type, param_array, n)
        f === nothing && return ERROR_INVALID_FUNCTION_TYPE
        
        # Solve
        S_min, min_val, x, iterations = fujishige_wolfe_submodular_minimization(
            f; ε=tolerance, max_iterations=Int(max_iterations), verbose=false
        )
        
        # Find indices of selected elements
        selected_indices = findall(S_min)
        set_size = length(selected_indices)
        
        # Allocate result array in Julia (1-indexed converted to 0-indexed for C)
        if set_size > 0
            result_array = [Cint(idx - 1) for idx in selected_indices]  # Convert to 0-indexed
        else
            result_array = Cint[]
        end
        
        # Store results
        unsafe_store!(result_value, Cdouble(min_val))
        unsafe_store!(result_iterations, Cint(iterations))
        
        # Return pointer to Julia-allocated array
        if set_size > 0
            unsafe_store!(result_set, pointer(result_array))
        else
            unsafe_store!(result_set, C_NULL)
        end
        
        return Cint(set_size)
        
    catch e
        @warn "Error in fujishige_wolfe_solve_c: $e"
        return ERROR_CONVERGENCE_FAILED
    end
end

"""
    wolfe_algorithm_c(func_type, params, n_params, n, tolerance, max_iterations, 
                      result_x, result_iterations, converged)

Direct Wolfe algorithm solver (returns minimum norm point, not minimizing set).

# Arguments
- `func_type::Cint`: Function type constant
- `params::Ptr{Cdouble}`: Pointer to parameter array
- `n_params::Cint`: Number of parameters
- `n::Cint`: Ground set size
- `tolerance::Cdouble`: Convergence tolerance
- `max_iterations::Cint`: Maximum iterations
- `result_x::Ptr{Ptr{Cdouble}}`: Output pointer to minimum norm point array (allocated by Julia)
- `result_iterations::Ptr{Cint}`: Output pointer for iteration count
- `converged::Ptr{Cint}`: Output pointer for convergence flag

# Returns
- `Cint`: SUCCESS (0) or error code (< 0)
"""
Base.@ccallable function wolfe_algorithm_c(
    func_type::Cint,
    params::Ptr{Cdouble},
    n_params::Cint,
    n::Cint,
    tolerance::Cdouble,
    max_iterations::Cint,
    result_x::Ptr{Ptr{Cdouble}},
    result_iterations::Ptr{Cint},
    converged::Ptr{Cint}
)::Cint
    try
        # Extract parameters from C array
        param_array = unsafe_wrap(Array, params, Int(n_params))
        
        # Create function
        f = create_function_from_params(func_type, param_array, n)
        f === nothing && return ERROR_INVALID_FUNCTION_TYPE
        
        # Run Wolfe algorithm
        x, iterations, has_converged = wolfe_algorithm(
            f; ε=tolerance, max_iterations=Int(max_iterations), verbose=false
        )
        
        # Allocate result array in Julia
        result_array = [Cdouble(xi) for xi in x]
        
        # Store results
        unsafe_store!(result_iterations, Cint(iterations))
        unsafe_store!(converged, has_converged ? Cint(1) : Cint(0))
        unsafe_store!(result_x, pointer(result_array))
        
        return SUCCESS
        
    catch e
        @warn "Error in wolfe_algorithm_c: $e"
        return ERROR_CONVERGENCE_FAILED
    end
end

"""
    check_submodular_c(func_type, params, n_params, n, violations)

Check if a function is submodular.

# Returns
- `Cint`: 1 if submodular, 0 if not, negative for error
"""
Base.@ccallable function check_submodular_c(
    func_type::Cint,
    params::Ptr{Cdouble},
    n_params::Cint,
    n::Cint,
    violations::Ptr{Cint}
)::Cint
    try
        # Extract parameters
        param_array = unsafe_wrap(Array, params, Int(n_params))
        
        # Create function
        f = create_function_from_params(func_type, param_array, n)
        f === nothing && return ERROR_INVALID_FUNCTION_TYPE
        
        # Check submodularity
        is_sub, violation_count, total_tests = is_submodular(f; verbose=false)
        
        # Store violation count
        unsafe_store!(violations, Cint(violation_count))
        
        return is_sub ? Cint(1) : Cint(0)
        
    catch e
        @warn "Error in check_submodular_c: $e"
        return ERROR_INVALID_PARAMETERS
    end
end

"""
    is_minimiser_c(func_type, params, n_params, n, candidate_set, set_size, improvement_value)

Check if a given set is optimal.

# Returns
- `Cint`: 1 if optimal, 0 if not, negative for error
"""
Base.@ccallable function is_minimiser_c(
    func_type::Cint,
    params::Ptr{Cdouble},
    n_params::Cint,
    n::Cint,
    candidate_set::Ptr{Cint},
    set_size::Cint,
    improvement_value::Ptr{Cdouble}
)::Cint
    try
        # Extract parameters
        param_array = unsafe_wrap(Array, params, Int(n_params))
        
        # Create function
        f = create_function_from_params(func_type, param_array, n)
        f === nothing && return ERROR_INVALID_FUNCTION_TYPE
        
        # Create BitVector from candidate set (convert from 0-indexed to 1-indexed)
        S = falses(Int(n))
        if set_size > 0
            indices = unsafe_wrap(Array, candidate_set, Int(set_size))
            for idx in indices
                if 1 <= idx + 1 <= n  # Convert from 0-indexed to 1-indexed
                    S[idx + 1] = true
                end
            end
        end
        
        # Check optimality
        is_optimal, improvement, better_val = is_minimiser(S, f; verbose=false)
        
        # Store improvement value
        if !is_optimal && !isnan(better_val)
            unsafe_store!(improvement_value, Cdouble(better_val))
        else
            unsafe_store!(improvement_value, Cdouble(0.0))
        end
        
        return is_optimal ? Cint(1) : Cint(0)
        
    catch e
        @warn "Error in is_minimiser_c: $e"
        return ERROR_INVALID_PARAMETERS
    end
end

"""
    get_function_type_constants()

Return pointers to the function type constants for Python to access.
"""
Base.@ccallable function get_function_type_constants(
    concave::Ptr{Cint},
    cut::Ptr{Cint},
    sqrt::Ptr{Cint},
    matroid::Ptr{Cint},
    bipartite::Ptr{Cint},
    facility::Ptr{Cint}
)::Cint
    try
        unsafe_store!(concave, FUNC_TYPE_CONCAVE)
        unsafe_store!(cut, FUNC_TYPE_CUT)
        unsafe_store!(sqrt, FUNC_TYPE_SQRT)
        unsafe_store!(matroid, FUNC_TYPE_MATROID)
        unsafe_store!(bipartite, FUNC_TYPE_BIPARTITE_MATCHING)
        unsafe_store!(facility, FUNC_TYPE_FACILITY_LOCATION)
        return SUCCESS
    catch e
        @warn "Error in get_function_type_constants: $e"
        return ERROR_MEMORY_ALLOCATION
    end
end

"""
    register_python_callback(callback_ptr)

Register a Python callback function for use with callback submodular functions.
The callback should have signature: double callback(int* indices, int n_indices)
"""
Base.@ccallable function register_python_callback(callback_ptr::Ptr{Cvoid})::Cint
    try
        PYTHON_CALLBACK[] = callback_ptr
        return SUCCESS
    catch e
        @warn "Error registering Python callback: $e"
        return ERROR_MEMORY_ALLOCATION
    end
end

"""
    test_python_callback(indices, n_indices, expected_result)

Test function to verify Python callback is working correctly.
"""
Base.@ccallable function test_python_callback(
    indices::Ptr{Cint}, 
    n_indices::Cint, 
    expected_result::Ptr{Cdouble}
)::Cint
    try
        if PYTHON_CALLBACK[] != C_NULL
            result = ccall(PYTHON_CALLBACK[], Cdouble, (Ptr{Cint}, Cint), indices, n_indices)
            unsafe_store!(expected_result, result)
            return SUCCESS
        else
            return ERROR_INVALID_PARAMETERS
        end
    catch e
        @warn "Error testing Python callback: $e"
        return ERROR_CONVERGENCE_FAILED
    end
end

"""
    clear_memoization_cache()

Clear the memoization cache for Python callbacks.
"""
Base.@ccallable function clear_memoization_cache()::Cint
    try
        empty!(MEMOIZATION_CACHE)
        CACHE_HITS[] = 0
        CACHE_MISSES[] = 0
        return SUCCESS
    catch e
        @warn "Error clearing memoization cache: $e"
        return ERROR_MEMORY_ALLOCATION
    end
end

"""
    get_cache_stats(cache_hits, cache_misses, cache_size)

Get memoization cache statistics.
"""
Base.@ccallable function get_cache_stats(
    cache_hits::Ptr{Cint},
    cache_misses::Ptr{Cint}, 
    cache_size::Ptr{Cint}
)::Cint
    try
        unsafe_store!(cache_hits, Cint(CACHE_HITS[]))
        unsafe_store!(cache_misses, Cint(CACHE_MISSES[]))
        unsafe_store!(cache_size, Cint(length(MEMOIZATION_CACHE)))
        return SUCCESS
    catch e
        @warn "Error getting cache stats: $e"
        return ERROR_MEMORY_ALLOCATION
    end
end

"""
    set_cache_enabled(enabled)

Enable or disable memoization caching.
"""
Base.@ccallable function set_cache_enabled(enabled::Cint)::Cint
    try
        # For now, cache is always enabled, but we could add a flag here
        # This function is provided for future extensibility
        return SUCCESS
    catch e
        @warn "Error setting cache enabled: $e"
        return ERROR_MEMORY_ALLOCATION
    end
end

"""
    free_result_array(ptr)

Free memory allocated by Julia for result arrays.
"""
Base.@ccallable function free_result_array(ptr::Ptr{Cint})::Cint
    # In Julia, we don't need to explicitly free arrays allocated by Julia
    # The GC will handle this, but we provide this function for API completeness
    return SUCCESS
end