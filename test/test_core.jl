"""
Tests for core functionality (abstract types and interfaces)
"""

@testset "Core Interface Tests" begin
    
    @testset "SubmodularFunction Abstract Type" begin
        # Test that SubmodularFunction is properly defined as abstract
        @test SubmodularFunction isa Type
        @test isabstracttype(SubmodularFunction)
        
        # Test that we cannot instantiate abstract type directly
        @test_throws MethodError SubmodularFunction()
    end
    
    @testset "Core Function Interface" begin
        # Test that required methods exist (they are function objects in the module)
        @test isdefined(SubmodularMinimization, :evaluate)
        @test isdefined(SubmodularMinimization, :ground_set_size)
        
        # Test with a simple concrete implementation for interface testing
        struct TestSubmodularFunction <: SubmodularFunction
            n::Int
        end
        
        SubmodularMinimization.ground_set_size(f::TestSubmodularFunction) = f.n
        SubmodularMinimization.evaluate(f::TestSubmodularFunction, S::BitVector) = sum(S)
        
        test_f = TestSubmodularFunction(5)
        
        @test ground_set_size(test_f) == 5
        @test evaluate(test_f, falses(5)) == 0
        @test evaluate(test_f, trues(5)) == 5
        @test evaluate(test_f, BitVector([1,0,1,0,1])) == 3
    end
    
    @testset "BitVector Operations" begin
        # Test BitVector creation and manipulation
        n = 8
        S = falses(n)
        @test length(S) == n
        @test sum(S) == 0
        
        S = trues(n)
        @test sum(S) == n
        
        # Test specific patterns
        S = BitVector([1,0,1,0,1,0,1,0])
        @test sum(S) == 4
        @test S[1] == true
        @test S[2] == false
        
        # Test XOR operations (used in cut functions)
        S1 = BitVector([1,0,1,0])
        S2 = BitVector([0,1,1,0]) 
        @test S1[1] ⊻ S2[1] == true   # 1 XOR 0 = true
        @test S1[2] ⊻ S2[2] == true   # 0 XOR 1 = true  
        @test S1[3] ⊻ S2[3] == false  # 1 XOR 1 = false
        @test S1[4] ⊻ S2[4] == false  # 0 XOR 0 = false
    end
end