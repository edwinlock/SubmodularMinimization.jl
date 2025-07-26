"""
Algorithm verification tests using is_minimiser().

This test suite uses the is_minimiser() function to verify that both
the brute force algorithm and the main Fujishige-Wolfe algorithm
correctly find optimal solutions.
"""

using Test
using SubmodularMinimization
using Random
using Statistics

# Set random seed for reproducibility
Random.seed!(42)

@testset "Algorithm Verification using is_minimiser()" begin
    
    @testset "Brute Force Verification" begin
        println("Verifying brute force algorithm correctness...")
        
        @testset "Known Submodular Functions" begin
            test_functions = [
                ("ConcaveSubmodularFunction", [ConcaveSubmodularFunction(n, 0.5) for n in 4:7]),
                ("SquareRootFunction", [SquareRootFunction(n) for n in 4:7]),
                ("MatroidRankFunction", [MatroidRankFunction(n, div(n,2)) for n in 4:7]),
            ]
            
            for (func_name, functions) in test_functions
                println("  Testing $func_name...")
                
                for f in functions
                    n = ground_set_size(f)
                    
                    # Run brute force
                    S_bf, val_bf = brute_force_minimization(f)
                    
                    # Verify the result is optimal using is_minimiser
                    is_opt, improvement, better_val = is_minimiser(S_bf, f)
                    @test is_opt == true
                    @test improvement == ""
                    @test isnan(better_val)
                    
                    # Additional check: verify no other set has a better value
                    @test val_bf <= 0.01  # For these functions, minimum should be ~0
                end
            end
        end
        
        @testset "Cut Functions" begin
            println("  Testing CutFunction...")
            
            # Test various graph structures
            test_graphs = [
                ("Path", [(1,2), (2,3), (3,4)], 4),
                ("Triangle", [(1,2), (1,3), (2,3)], 3),
                ("Star", [(1,2), (1,3), (1,4)], 4),
                ("Complete_4", [(i,j) for i in 1:4 for j in (i+1):4], 4)
            ]
            
            for (graph_name, edges, n) in test_graphs
                f = CutFunction(n, edges)
                
                S_bf, val_bf = brute_force_minimization(f)
                is_opt, improvement, better_val = is_minimiser(S_bf, f)
                
                @test is_opt == true
                @test val_bf >= 0  # Cut values are non-negative
                
                # For most graphs, minimum cut should be 0 (empty or full set)
                if graph_name in ["Path", "Triangle", "Star"]
                    @test abs(val_bf) < 1e-10
                end
            end
        end
        
        @testset "Random Functions" begin
            println("  Testing random submodular functions...")
            
            for trial in 1:20
                n = rand(4:6)
                func_type = rand(1:4)
                
                if func_type == 1
                    f = ConcaveSubmodularFunction(n, 0.2 + rand() * 0.6)
                elseif func_type == 2
                    f = SquareRootFunction(n)
                elseif func_type == 3
                    f = MatroidRankFunction(n, rand(1:(n-1)))
                else
                    f = create_random_cut_function(n, 0.3 + rand() * 0.4)
                end
                
                S_bf, val_bf = brute_force_minimization(f)
                is_opt, improvement, better_val = is_minimiser(S_bf, f)
                
                @test is_opt == true
                if !is_opt
                    @warn "Brute force failed for function type $func_type, n=$n: $improvement"
                end
            end
        end
    end
    
    @testset "Main Algorithm Verification" begin
        println("Verifying Fujishige-Wolfe algorithm correctness...")
        
        @testset "Known Submodular Functions" begin
            success_counts = Dict{String, Int}()
            total_counts = Dict{String, Int}()
            
            test_functions = [
                ("ConcaveSubmodularFunction", [ConcaveSubmodularFunction(n, 0.1 + rand() * 0.8) for n in 4:8 for _ in 1:5]),
                ("SquareRootFunction", [SquareRootFunction(n) for n in 4:8]),
                ("MatroidRankFunction", [MatroidRankFunction(n, rand(1:(n-1))) for n in 4:8 for _ in 1:3]),
            ]
            
            for (func_name, functions) in test_functions
                println("  Testing $func_name...")
                success_counts[func_name] = 0
                total_counts[func_name] = length(functions)
                
                for f in functions
                    n = ground_set_size(f)
                    
                    # Run main algorithm
                    S_alg, val_alg, _, iters = fujishige_wolfe_submodular_minimization(f; verbose=false)
                    
                    # Verify the result is optimal using is_minimiser
                    is_opt, improvement, better_val = is_minimiser(S_alg, f)
                    
                    if is_opt
                        success_counts[func_name] += 1
                    else
                        @warn "Main algorithm failed for $func_name (n=$n): $improvement. Got value $val_alg, better value $better_val"
                    end
                    
                    @test is_opt == true
                end
                
                success_rate = success_counts[func_name] / total_counts[func_name] * 100
                println("    Success rate: $(round(success_rate, digits=1))% ($(success_counts[func_name])/$(total_counts[func_name]))")
            end
        end
        
        @testset "Cut Functions" begin
            println("  Testing CutFunction...")
            
            successes = 0
            total = 20
            
            for trial in 1:total
                n = rand(4:7)
                edge_prob = 0.2 + rand() * 0.5
                f = create_random_cut_function(n, edge_prob)
                
                S_alg, val_alg, _, _ = fujishige_wolfe_submodular_minimization(f; verbose=false)
                is_opt, improvement, better_val = is_minimiser(S_alg, f)
                
                if is_opt
                    successes += 1
                else
                    @warn "Cut function failed (trial $trial): $improvement"
                end
            end
            
            success_rate = successes / total * 100
            println("    Cut function success rate: $(round(success_rate, digits=1))% ($successes/$total)")
            @test successes == total  # Should be 100% for submodular functions
        end
        
        @testset "Advanced Submodular Functions" begin
            println("  Testing advanced submodular functions...")
            
            advanced_functions = [
                ("FacilityLocation", [create_random_facility_location(rand(4:6), rand(3:5)) for _ in 1:10]),
                ("WeightedCoverage", [create_random_coverage_function(rand(4:6), rand(4:8); coverage_prob=0.3) for _ in 1:10]),
                ("LogDeterminant", [create_wishart_log_determinant(rand(3:5)) for _ in 1:5]),
                ("Entropy", [create_random_entropy_function(rand(4:6), rand(2:4)) for _ in 1:10]),
            ]
            
            for (func_name, functions) in advanced_functions
                println("    Testing $func_name...")
                successes = 0
                
                for f in functions
                    S_alg, val_alg, _, _ = fujishige_wolfe_submodular_minimization(f; verbose=false)
                    is_opt, improvement, better_val = is_minimiser(S_alg, f)
                    
                    if is_opt
                        successes += 1
                    else
                        @warn "$func_name failed: $improvement"
                    end
                end
                
                success_rate = successes / length(functions) * 100
                println("      Success rate: $(round(success_rate, digits=1))% ($successes/$(length(functions)))")
                @test successes == length(functions)  # Should be 100% for submodular functions
            end
        end
    end
    
    @testset "Direct Algorithm Comparison" begin
        println("Comparing brute force vs main algorithm...")
        
        @testset "Small Problems - Exact Comparison" begin
            test_functions = [
                ConcaveSubmodularFunction(5, 0.7),
                SquareRootFunction(5),
                MatroidRankFunction(5, 2),
                CutFunction(5, [(1,2), (2,3), (3,4), (4,5)]),
                create_random_facility_location(5, 3),
            ]
            
            for f in test_functions
                # Run both algorithms
                S_bf, val_bf = brute_force_minimization(f)
                S_alg, val_alg, _, _ = fujishige_wolfe_submodular_minimization(f; verbose=false)
                
                # Both results should be optimal
                is_opt_bf, improvement_bf, _ = is_minimiser(S_bf, f)
                is_opt_alg, improvement_alg, _ = is_minimiser(S_alg, f)
                
                @test is_opt_bf == true
                @test is_opt_alg == true
                
                # Function values should be the same (within tolerance)
                @test abs(val_bf - val_alg) < COMPARISON_TOLERANCE
                
                if abs(val_bf - val_alg) > COMPARISON_TOLERANCE
                    @warn "Value mismatch: BF=$val_bf, ALG=$val_alg, diff=$(abs(val_bf - val_alg))"
                end
            end
        end
        
        @testset "Statistical Comparison" begin
            println("  Running statistical comparison...")
            
            # Test many random instances
            differences = Float64[]
            bf_failures = 0
            alg_failures = 0
            total_tests = 50
            
            for trial in 1:total_tests
                n = rand(4:6)
                f = ConcaveSubmodularFunction(n, 0.3 + rand() * 0.5)
                
                # Run both algorithms
                S_bf, val_bf = brute_force_minimization(f)
                S_alg, val_alg, _, _ = fujishige_wolfe_submodular_minimization(f; verbose=false)
                
                # Check optimality
                is_opt_bf, _, _ = is_minimiser(S_bf, f)
                is_opt_alg, _, _ = is_minimiser(S_alg, f)
                
                if !is_opt_bf
                    bf_failures += 1
                end
                if !is_opt_alg
                    alg_failures += 1
                end
                
                push!(differences, abs(val_bf - val_alg))
            end
            
            println("    Brute force failures: $bf_failures / $total_tests")
            println("    Algorithm failures: $alg_failures / $total_tests")
            println("    Mean value difference: $(round(mean(differences), digits=8))")
            println("    Max value difference: $(round(maximum(differences), digits=8))")
            println("    Std value difference: $(round(std(differences), digits=8))")
            
            @test bf_failures == 0  # Brute force should never fail
            @test alg_failures == 0 # Main algorithm should never fail on submodular functions
            @test maximum(differences) < COMPARISON_TOLERANCE
        end
    end
    
    @testset "Non-Submodular Function Behavior" begin
        println("Testing behavior on non-submodular functions...")
        
        @testset "Feature Selection Functions" begin
            println("  Testing Feature Selection functions...")
            
            local_optimal_count = 0
            global_optimal_count = 0
            total_tests = 20
            
            for trial in 1:total_tests
                n = rand(4:6)
                f = create_feature_selection(n)
                
                # Run main algorithm (may find local minimum)
                S_alg, val_alg, _, _ = fujishige_wolfe_submodular_minimization(f; verbose=false)
                
                # Run brute force (finds global minimum)
                S_bf, val_bf = brute_force_minimization(f)
                
                # Check if algorithm result is locally optimal
                is_local_opt, improvement_alg, _ = is_minimiser(S_alg, f)
                is_global_opt, improvement_bf, _ = is_minimiser(S_bf, f)
                
                if is_local_opt
                    local_optimal_count += 1
                end
                
                if is_global_opt
                    global_optimal_count += 1
                end
                
                # Brute force should always find globally optimal solution
                @test is_global_opt == true
                
                # Algorithm should find locally optimal solution (but may not for non-submodular functions)
                # @test is_local_opt == true  # Commented out since Feature Selection functions are non-submodular
                
                # Check if they found the same solution
                if abs(val_alg - val_bf) > 1e-6
                    println("    Trial $trial: Local min = $(round(val_alg, digits=6)), Global min = $(round(val_bf, digits=6))")
                end
            end
            
            println("    Local optimal solutions found: $local_optimal_count / $total_tests")
            println("    Global optimal solutions found: $global_optimal_count / $total_tests")
            
            # Brute force should always be globally optimal
            @test global_optimal_count == total_tests
            
            # For non-submodular functions, we expect some algorithms to not find locally optimal solutions
            # @test local_optimal_count == total_tests  # Commented out - expected to fail for non-submodular functions
        end
    end
    
    @testset "Consistency Checks" begin
        println("Running consistency checks...")
        
        @testset "Optimality implies minimum value" begin
            # If is_minimiser says a solution is optimal, it should have the minimum value
            
            for trial in 1:10
                n = rand(4:6)
                f = ConcaveSubmodularFunction(n, 0.5)
                
                # Find minimum with brute force
                S_bf, val_bf = brute_force_minimization(f)
                
                # Test multiple random sets
                for _ in 1:5
                    S_random = rand(n) .< 0.3  # Mostly small sets
                    is_opt, improvement, better_val = is_minimiser(S_random, f)
                    val_random = evaluate(f, S_random)
                    
                    if is_opt
                        # If optimal, value should equal minimum
                        @test abs(val_random - val_bf) < COMPARISON_TOLERANCE
                    else
                        # If not optimal, should have worse value
                        @test val_random > val_bf - COMPARISON_TOLERANCE
                        @test better_val < val_random - COMPARISON_TOLERANCE
                    end
                end
            end
        end
        
        @testset "Improvement suggestions are valid" begin
            # When is_minimiser suggests an improvement, it should actually be better
            
            for trial in 1:20
                n = rand(4:6)
                f = ConcaveSubmodularFunction(n, 0.7)
                
                # Generate a likely suboptimal set
                S = rand(n) .< 0.7  # Larger sets are suboptimal for concave functions
                
                is_opt, improvement, better_val = is_minimiser(S, f)
                current_val = evaluate(f, S)
                
                if !is_opt
                    # The suggested improvement should actually be better
                    @test better_val < current_val - COMPARISON_TOLERANCE
                    
                    # Parse the improvement message to verify
                    if contains(improvement, "adding element")
                        element = parse(Int, match(r"adding element (\d+)", improvement).captures[1])
                        S_improved = copy(S)
                        S_improved[element] = true
                        @test abs(evaluate(f, S_improved) - better_val) < 1e-10
                        
                    elseif contains(improvement, "removing element")
                        element = parse(Int, match(r"removing element (\d+)", improvement).captures[1])
                        S_improved = copy(S)
                        S_improved[element] = false
                        @test abs(evaluate(f, S_improved) - better_val) < 1e-10
                    end
                end
            end
        end
    end
    
    println("All algorithm verification tests completed!")
end