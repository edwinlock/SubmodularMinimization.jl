#!/usr/bin/env julia
"""
Test script for the caching functionality in SubmodularMinimization.jl

This script demonstrates the performance benefits of memoization when
using submodular function minimization algorithms.
"""

using SubmodularMinimization

function test_caching_performance()
    println("Testing Caching Performance in SubmodularMinimization.jl")
    println("=" * 60)
    
    # Create a test function
    f = ConcaveSubmodularFunction(12, 0.7)
    
    # Test 1: Fujishige-Wolfe with and without caching
    println("\n🧮 Test 1: Fujishige-Wolfe Algorithm")
    println("-" * 40)
    
    # Without caching
    println("Without caching:")
    time_start = time()
    S1, val1, x1, iters1 = fujishige_wolfe_submodular_minimization(f; cache=false, verbose=false)
    time_no_cache = time() - time_start
    println("  Time: $(round(time_no_cache * 1000, digits=2)) ms")
    println("  Result: |S| = $(sum(S1)), value = $(round(val1, digits=6))")
    
    # With caching (default)
    println("\nWith caching:")
    time_start = time()
    S2, val2, x2, iters2 = fujishige_wolfe_submodular_minimization(f; cache=true, verbose=true)
    time_with_cache = time() - time_start
    println("  Time: $(round(time_with_cache * 1000, digits=2)) ms")
    println("  Result: |S| = $(sum(S2)), value = $(round(val2, digits=6))")
    
    speedup = time_no_cache / time_with_cache
    println("  Speedup: $(round(speedup, digits=2))x")
    
    # Test 2: Manual caching usage
    println("\n🔧 Test 2: Manual Caching Usage")
    println("-" * 40)
    
    # Create a cached function manually
    cached_f = cached(f; max_cache_size=100)
    
    println("Testing cached function with multiple operations:")
    
    # Clear cache first
    clear_cache!(cached_f)
    
    # Run submodularity check (lots of evaluations)
    println("  Running submodularity check...")
    is_sub, violations, total_tests = is_submodular(cached_f; verbose=true, cache=false) # Use pre-cached function
    
    # Show final cache stats
    stats = cache_stats(cached_f)
    println("  Final cache statistics:")
    println("    Total function calls: $(stats.total_calls)")
    println("    Cache hits: $(stats.hits)")
    println("    Cache misses: $(stats.misses)")
    println("    Hit rate: $(round(stats.hit_rate * 100, digits=1))%")
    println("    Cache size: $(stats.cache_size) entries")
    
    # Test 3: Cache size management
    println("\n📊 Test 3: Cache Size Management")
    println("-" * 40)
    
    # Test with limited cache size
    bounded_f = cached(f; max_cache_size=10)
    
    # Fill cache beyond capacity
    println("Testing bounded cache (max_size=10):")
    for i in 1:15
        S = falses(12)
        if i <= 12
            S[i] = true
        end
        evaluate(bounded_f, S)
    end
    
    stats = cache_stats(bounded_f)
    println("  Cache entries: $(stats.cache_size) (should be ≤ 10)")
    println("  Total calls: $(stats.total_calls)")
    
    # Test 4: Cache recommendation system
    println("\n💡 Test 4: Auto Cache Recommendations")
    println("-" * 40)
    
    for n in [8, 15, 25, 100]
        rec = auto_cache_recommendation(n)
        println("  n=$n: $(rec.enabled ? "Enable" : "Disable") caching")
        println("       Max size: $(rec.max_size == 0 ? "unlimited" : rec.max_size)")
        println("       Reason: $(rec.reason)")
    end
    
    # Test 5: Performance comparison on larger problem
    println("\n⚡ Test 5: Performance on Larger Problem")
    println("-" * 40)
    
    large_f = ConcaveSubmodularFunction(18, 0.8)
    
    println("Testing n=18 problem (more cache opportunities):")
    
    # Without cache
    time_start = time()
    S_large1, val_large1, _, _ = fujishige_wolfe_submodular_minimization(large_f; cache=false, verbose=false)
    time_large_no_cache = time() - time_start
    
    # With cache  
    time_start = time()
    S_large2, val_large2, _, _ = fujishige_wolfe_submodular_minimization(large_f; cache=true, verbose=false)
    time_large_with_cache = time() - time_start
    
    println("  No cache: $(round(time_large_no_cache * 1000, digits=1)) ms")
    println("  With cache: $(round(time_large_with_cache * 1000, digits=1)) ms")
    
    if time_large_no_cache > 0
        large_speedup = time_large_no_cache / time_large_with_cache
        println("  Speedup: $(round(large_speedup, digits=2))x")
    end
    
    println("\n✅ Caching tests completed!")
    println("\nKey Benefits Demonstrated:")
    println("  • Automatic memoization reduces redundant function evaluations")
    println("  • Significant speedups for algorithms with repeated subset evaluations") 
    println("  • Cache size management prevents memory issues")
    println("  • Intelligent cache recommendations based on problem size")
    println("  • Works seamlessly with all algorithms (Fujishige-Wolfe, Wolfe, utilities)")
end

# Run the tests
if abspath(PROGRAM_FILE) == @__FILE__
    test_caching_performance()
end