"""
Caching Infrastructure for SubmodularMinimization.jl

This module provides memoization capabilities for submodular functions to reduce
redundant function evaluations during optimization algorithms.
"""

using LinearAlgebra

"""
    CachedSubmodularFunction <: SubmodularFunction

A wrapper that adds memoization to any submodular function.

This wrapper caches function evaluations to avoid redundant calls during optimization.
The cache uses BitVector keys for exact subset matching.

# Fields
- `f::SubmodularFunction`: The underlying function to cache
- `cache::Dict{BitVector, Float64}`: Cache storage mapping subsets to values
- `hits::Int`: Number of cache hits
- `misses::Int`: Number of cache misses
- `max_cache_size::Int`: Maximum cache size (0 = unlimited)
"""
mutable struct CachedSubmodularFunction <: SubmodularFunction
    f::SubmodularFunction
    cache::Dict{BitVector, Float64}
    hits::Int
    misses::Int
    max_cache_size::Int
    
    function CachedSubmodularFunction(f::SubmodularFunction; max_cache_size::Int=0)
        new(f, Dict{BitVector, Float64}(), 0, 0, max_cache_size)
    end
end

# Pass through the ground set size
ground_set_size(cf::CachedSubmodularFunction) = ground_set_size(cf.f)

"""
    evaluate(cf::CachedSubmodularFunction, S::BitVector)

Evaluate the cached function, using cache when possible.
"""
function evaluate(cf::CachedSubmodularFunction, S::BitVector)
    # Check cache first
    if haskey(cf.cache, S)
        cf.hits += 1
        return cf.cache[S]
    end
    
    # Cache miss - evaluate underlying function
    cf.misses += 1
    result = evaluate(cf.f, S)
    
    # Store in cache (with size management if needed)
    if cf.max_cache_size > 0 && length(cf.cache) >= cf.max_cache_size
        # Simple eviction: remove oldest entry (first in iteration order)
        first_key = first(keys(cf.cache))
        delete!(cf.cache, first_key)
    end
    
    cf.cache[copy(S)] = result
    return result
end

"""
    cache_stats(cf::CachedSubmodularFunction)

Get cache performance statistics.

# Returns
A named tuple with:
- `hits`: Number of cache hits
- `misses`: Number of cache misses  
- `cache_size`: Current number of cached entries
- `hit_rate`: Cache hit rate (0.0 to 1.0)
- `total_calls`: Total function evaluations
"""
function cache_stats(cf::CachedSubmodularFunction)
    total = cf.hits + cf.misses
    hit_rate = total > 0 ? cf.hits / total : 0.0
    
    return (
        hits = cf.hits,
        misses = cf.misses, 
        cache_size = length(cf.cache),
        hit_rate = hit_rate,
        total_calls = total
    )
end

"""
    clear_cache!(cf::CachedSubmodularFunction)

Clear the cache and reset statistics.
"""
function clear_cache!(cf::CachedSubmodularFunction)
    empty!(cf.cache)
    cf.hits = 0
    cf.misses = 0
    return cf
end

"""
    resize_cache!(cf::CachedSubmodularFunction, new_max_size::Int)

Change the maximum cache size. If new size is smaller, excess entries are evicted.
"""
function resize_cache!(cf::CachedSubmodularFunction, new_max_size::Int)
    cf.max_cache_size = new_max_size
    
    # Evict entries if cache is too large
    if new_max_size > 0
        while length(cf.cache) > new_max_size
            first_key = first(keys(cf.cache))
            delete!(cf.cache, first_key)
        end
    end
    
    return cf
end

"""
    cached(f::SubmodularFunction; max_cache_size::Int=0)

Convenience function to wrap a function with caching.

# Arguments
- `f`: The submodular function to cache
- `max_cache_size`: Maximum cache entries (0 = unlimited)

# Example
```julia
f = ConcaveSubmodularFunction(10, 0.7)
cached_f = cached(f)
result = fujishige_wolfe_submodular_minimization(cached_f)
println("Cache stats: ", cache_stats(cached_f))
```
"""
function cached(f::SubmodularFunction; max_cache_size::Int=0)
    return CachedSubmodularFunction(f; max_cache_size=max_cache_size)
end

"""
    enable_caching(f::SubmodularFunction; max_cache_size::Int=0)

Enable caching for a function (alias for `cached`).
"""
enable_caching(f::SubmodularFunction; max_cache_size::Int=0) = cached(f; max_cache_size=max_cache_size)

"""
    auto_cache_recommendation(n::Int)

Recommend cache settings based on problem size.

For small problems, unlimited caching is safe. For larger problems,
we recommend bounded cache sizes to manage memory usage.
"""
function auto_cache_recommendation(n::Int)
    if n <= 10
        return (enabled=true, max_size=0, reason="Small problem - unlimited cache safe")
    elseif n <= 20  
        return (enabled=true, max_size=1000, reason="Medium problem - bounded cache recommended")
    elseif n <= 50
        return (enabled=true, max_size=500, reason="Large problem - small cache recommended")
    else
        return (enabled=false, max_size=0, reason="Very large problem - caching may use too much memory")
    end
end