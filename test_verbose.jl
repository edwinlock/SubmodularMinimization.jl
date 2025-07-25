using Pkg
using Test, Dates

Pkg.activate(".")
using SubmodularMinimization


println("🚀 Starting Julia test suite...")
println("Time: $(Dates.now())")
println("=" ^ 50)

@testset "SubmodularMinimization.jl Tests" begin
    
    println("\n[1/6] 🧪 Running Core Interface Tests...")
    @time include("test/test_core.jl")
    
    println("\n[2/6] 🧪 Running Example Functions Tests...")
    @time include("test/test_examples.jl")
    
    println("\n[3/6] 🧪 Running Oracle Tests...")
    @time include("test/test_oracles.jl")
    
    println("\n[4/6] 🧪 Running Algorithm Tests...")
    @time include("test/test_algorithms.jl")
        
    println("\n[5/6] 🧪 Running Edge Case Tests...")
    @time include("test/test_edge_cases.jl")
    
    println("\n[6/6] 🧪 Running Correctness Comparison Tests...")
    @time include("test/test_correctness_comparison.jl")
end

println("\n🏁 All tests completed!")