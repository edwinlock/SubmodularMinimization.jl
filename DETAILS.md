# SubmodularMinimization.jl: Detailed Technical Documentation

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Algorithm Overview](#algorithm-overview)
3. [Implementation Architecture](#implementation-architecture)
4. [Core Data Structures](#core-data-structures)
5. [Algorithmic Components](#algorithmic-components)
6. [Numerical Stability](#numerical-stability)
7. [Performance Optimizations](#performance-optimizations)
8. [Memory Management](#memory-management)
9. [Testing Framework](#testing-framework)
10. [Example Functions](#example-functions)
11. [Advanced Usage](#advanced-usage)

---

## Theoretical Background

### Submodular Functions

A set function f: 2^V → ℝ is **submodular** if it satisfies the diminishing returns property:

```
f(S ∪ {v}) - f(S) ≥ f(T ∪ {v}) - f(T)
```

for all S ⊆ T ⊆ V and v ∉ T.

Equivalently, f is submodular if and only if:
```
f(S) + f(T) ≥ f(S ∪ T) + f(S ∩ T)
```

### The Submodular Minimization Problem

Given a submodular function f: 2^V → ℝ, we want to solve:
```
minimize f(S) subject to S ⊆ V
```

This problem is polynomial-time solvable (unlike submodular maximization, which is NP-hard).

### Base Polytope and Fujishige's Theorem

The **base polytope** B(f) of a submodular function f is:
```
B(f) = {x ∈ ℝ^n : x(S) ≤ f(S) for all S ⊆ V, x(V) = f(V)}
```

**Fujishige's Theorem**: The minimum of f is achieved by any set S such that χ_S = argmin_{x ∈ B(f)} ||x||_2^2, where χ_S is the characteristic vector of S.

This reduces submodular minimization to finding the minimum norm point in the base polytope.

### Wolfe's Algorithm

Wolfe's algorithm (1976) finds the minimum norm point in a polytope using the **Frank-Wolfe method**:

1. Start with an initial point x₀ in the polytope
2. At iteration k:
   - Find the vertex v_k that minimizes ⟨x_k, v_k⟩ (linear optimization oracle)
   - Compute step size α_k via line search
   - Update: x_{k+1} = x_k + α_k(v_k - x_k)
3. Continue until convergence

---

## Algorithm Overview

### High-Level Structure

```julia
function fujishige_wolfe_submodular_minimization(f::SubmodularFunction)
    # 1. Find minimum norm point in base polytope
    x, iterations, converged = wolfe_algorithm(f)
    
    # 2. Extract minimizing set using Fujishige's theorem
    S_min = extract_minimizing_set(x, f)
    
    # 3. Compute minimum value
    min_val = evaluate(f, S_min)
    
    return S_min, min_val, x, iterations
end
```

### The Wolfe Algorithm Implementation

```julia
function wolfe_algorithm!(workspace, f; ε=1e-6, max_iterations=10000)
    n = ground_set_size(f)
    
    # Initialize with first vertex from linear oracle
    s₁ = linear_optimization_oracle(f, zeros(n))
    workspace.S_matrix[:, 1] = s₁
    workspace.λ_buffer[1] = 1.0
    k = 1
    
    for iteration in 1:max_iterations
        # Current point: x = Σᵢ λᵢ sᵢ
        workspace.x_current .= workspace.S_matrix[:, 1:k] * workspace.λ_buffer[1:k]
        
        # Linear optimization oracle: find vertex minimizing ⟨x, s⟩
        s_new = linear_optimization_oracle!(workspace, f, workspace.x_current)
        
        # Check convergence: if ⟨x, s_new - x⟩ ≥ -ε, stop
        gap = dot(workspace.x_current, s_new - workspace.x_current)
        if gap ≥ -ε
            break
        end
        
        # Add new vertex to active set
        k += 1
        workspace.S_matrix[:, k] = s_new
        
        # Solve for optimal convex combination (affine minimization)
        workspace.λ_buffer[1:k] = affine_minimizer!(workspace, k)
        
        # Away-step and drop-step optimizations
        cleanup_vertices!(workspace, k)
    end
    
    return workspace.x_current, iteration, converged
end
```

---

## Implementation Architecture

### Module Structure

```
SubmodularMinimization.jl
├── src/
│   ├── SubmodularMinimization.jl    # Main module, exports
│   ├── core.jl                      # Abstract types and interfaces  
│   ├── oracles.jl                   # Linear oracle and affine minimization
│   ├── algorithms.jl                # Main Wolfe algorithm
│   ├── examples.jl                  # Concrete submodular functions
│   └── utils.jl                     # Testing and benchmarking utilities
└── test/                            # Comprehensive test suite
```

### Type Hierarchy

```julia
abstract type SubmodularFunction end

# Interface functions
ground_set_size(f::SubmodularFunction)::Int
evaluate(f::SubmodularFunction, S::BitVector)::Float64

# Concrete implementations
struct ConcaveSubmodularFunction <: SubmodularFunction
struct CutFunction <: SubmodularFunction
struct FacilityLocationFunction <: SubmodularFunction
# ... many more
```

---

## Core Data Structures

### WolfeWorkspace

The main workspace for pre-allocated algorithm execution:

```julia
mutable struct WolfeWorkspace
    n::Int                           # Ground set size
    max_vertices::Int                # Maximum vertices in active set
    
    # Linear optimization oracle buffers
    perm_buffer::Vector{Int}         # Permutation for greedy oracle
    S_buffer::BitVector              # Set representation buffer
    q_buffer::Vector{Float64}        # Marginal values buffer
    
    # Matrix storage for vertices (columns are vertices)
    S_matrix::Matrix{Float64}        # n × max_vertices matrix
    
    # Affine minimization workspace
    affine_workspace::AffineWorkspace
    y_buffer::Vector{Float64}        # Objective gradients
    α_buffer::Vector{Float64}        # Step sizes
    
    # Convex combination coefficients
    λ_buffer::Vector{Float64}        # λᵢ weights for vertices
    
    # Algorithm state vectors
    x_temp::Vector{Float64}          # Temporary computations
    x_current::Vector{Float64}       # Current iterate x_k
    θ_values::Vector{Float64}        # Line search parameters
    
    # Index management
    keep_indices_buffer::Vector{Int} # Active vertex indices
    keep_count::Int                  # Number of active vertices
    
    # Numerical stability tracking
    marginal_stats::Union{Nothing, Tuple{Float64, Float64}}
end
```

### AffineWorkspace

Specialized workspace for affine minimization subproblems:

```julia
mutable struct AffineWorkspace
    n::Int                           # Dimension
    max_dim::Int                     # Maximum active set size
    
    # QR decomposition workspace
    Q::Matrix{Float64}               # Q matrix (n × max_dim)
    R::Matrix{Float64}               # R matrix (max_dim × max_dim)
    τ::Vector{Float64}               # Householder reflectors
    work::Vector{Float64}            # LAPACK workspace
    
    # Linear system solving
    A_active::Matrix{Float64}        # Active constraint matrix
    b_active::Vector{Float64}        # Active RHS vector
    x_solution::Vector{Float64}      # Solution vector
    
    # Constraint management
    active_constraints::Vector{Int}  # Active constraint indices
    constraint_violations::Vector{Float64} # Violation measures
end
```

---

## Algorithmic Components

### 1. Linear Optimization Oracle

The linear optimization oracle solves:
```
minimize ⟨c, s⟩ subject to s ∈ vertices(B(f))
```

For submodular functions, vertices of the base polytope are **tight sets**: sets S where the inequality x(S) ≤ f(S) is tight.

**Greedy Algorithm**:
```julia
function linear_optimization_oracle!(workspace, f, c)
    n = length(c)
    
    # Compute marginal values: q[i] = f({i}) - f(∅) for sorting
    for i in 1:n
        workspace.S_buffer .= false
        workspace.S_buffer[i] = true
        workspace.q_buffer[i] = evaluate(f, workspace.S_buffer)
    end
    
    # Sort elements by c[i] - q[i] in ascending order
    sortperm!(workspace.perm_buffer, c - workspace.q_buffer)
    
    # Greedily construct tight set
    workspace.S_buffer .= false
    current_value = 0.0
    s = zeros(n)
    
    for idx in workspace.perm_buffer
        workspace.S_buffer[idx] = true
        new_value = evaluate(f, workspace.S_buffer)
        s[idx] = new_value - current_value
        current_value = new_value
    end
    
    return s
end
```

**Key Insight**: The greedy algorithm produces a vertex of the base polytope in O(n log n + nT) time, where T is the function evaluation time.

### 2. Affine Minimization

At each iteration, we solve the constrained quadratic program:
```
minimize (1/2)||Σᵢ λᵢ sᵢ||² subject to Σᵢ λᵢ = 1, λᵢ ≥ 0
```

This is solved using **active set methods** with QR decomposition:

```julia
function affine_minimizer!(workspace, k)
    S = workspace.S_matrix[:, 1:k]
    
    # Form Gram matrix: G[i,j] = ⟨sᵢ, sⱼ⟩
    G = S' * S
    
    # Solve: minimize (1/2)λᵀGλ subject to eᵀλ = 1, λ ≥ 0
    # This is a convex quadratic program
    
    # Active set method with warm start
    λ = solve_constrained_qp(G, workspace.affine_workspace)
    
    return λ
end
```

**Numerical Considerations**:
- QR decomposition for stability
- Iterative refinement for ill-conditioned systems
- Pivot selection for numerical rank determination

### 3. Vertex Management

**Away Steps**: Remove vertices with λᵢ = 0 from the active set to maintain sparsity.

**Drop Steps**: When adding a new vertex would exceed storage, remove the least important vertex.

```julia
function cleanup_vertices!(workspace, k)
    # Remove vertices with near-zero coefficients
    tolerance = 1e-12
    active_count = 0
    
    for i in 1:k
        if workspace.λ_buffer[i] > tolerance
            active_count += 1
            if active_count != i
                # Compact storage
                workspace.S_matrix[:, active_count] = workspace.S_matrix[:, i]
                workspace.λ_buffer[active_count] = workspace.λ_buffer[i]
            end
        end
    end
    
    workspace.keep_count = active_count
    return active_count
end
```

### 4. Convergence Analysis

**Theoretical Guarantee**: Wolfe's algorithm converges at rate O(1/ε) for submodular functions.

**Practical Convergence**: The algorithm typically converges much faster due to:
- Strong convexity properties of the norm objective
- Good geometry of submodular base polytopes
- Adaptive step size selection

**Convergence Test**:
```julia
# Duality gap test
gap = dot(x_current, s_new - x_current)
converged = gap ≥ -ε
```

---

## Numerical Stability

### Sources of Numerical Error

1. **Function Evaluation**: Floating-point errors in submodular function computation
2. **Linear Algebra**: Conditioning of Gram matrices in affine minimization
3. **Accumulation**: Repeated floating-point operations in iterative algorithm

### Stability Enhancements

#### 1. Tolerance Management
```julia
const DEFAULT_TOLERANCE = 1e-6      # Balanced accuracy/speed
const LOOSE_TOLERANCE = 1e-4        # Fast convergence
const TIGHT_TOLERANCE = 5e-7        # High precision
const NUMERICAL_PRECISION_TOLERANCE = 1e-12  # Machine precision level
```

#### 2. Marginal Value Monitoring
```julia
function track_marginal_stability!(workspace, f, x)
    # Compute marginal value statistics for stability detection
    marginals = [marginal_value(f, x, i) for i in 1:length(x)]
    min_marginal = minimum(marginals)
    max_marginal = maximum(marginals)
    
    workspace.marginal_stats = (min_marginal, max_marginal)
    
    # Warn if marginals are poorly conditioned
    if max_marginal - min_marginal > 1e8
        @warn "Marginal values poorly conditioned: range $(max_marginal - min_marginal)"
    end
end
```

#### 3. QR Decomposition with Pivoting
```julia
function solve_constrained_qp_stable(G, workspace)
    # Use pivoted QR for rank-deficient matrices
    F = qr(G, ColumnNorm())
    
    # Determine numerical rank
    diagonal_R = diag(F.R)
    numerical_rank = sum(abs.(diagonal_R) .> 1e-12 * abs(diagonal_R[1]))
    
    # Solve reduced system
    if numerical_rank < size(G, 1)
        @warn "Rank deficient Gram matrix detected"
        # Use regularization or pseudoinverse
    end
    
    return solve_with_rank(F, numerical_rank, workspace)
end
```

#### 4. Iterative Refinement
```julia
function iterative_refinement!(x, A, b, tolerance=1e-14)
    # Newton-style iterative refinement for improved accuracy
    for iter in 1:3
        residual = A * x - b
        if norm(residual) < tolerance
            break
        end
        correction = A \ residual
        x -= correction
    end
    return x
end
```

---

## Performance Optimizations

### 1. Memory Layout Optimization

**Column-Major Matrix Storage**: Vertices stored as columns for cache-friendly access:
```julia
# Efficient: access S_matrix[:, i] (column)
S_matrix = Matrix{Float64}(undef, n, max_vertices)

# Less efficient: access S_matrix[i, :] (row)
```

**Buffer Reuse**: Pre-allocated buffers minimize garbage collection:
```julia
# Reuse buffers across iterations
workspace.perm_buffer  # Sorting permutations
workspace.S_buffer     # Set representations  
workspace.q_buffer     # Marginal values
```

### 2. BLAS/LAPACK Integration

**Level 3 BLAS Operations**: Matrix-matrix multiplies for Gram matrix computation:
```julia
# Compute G = S'S using BLAS
BLAS.syrk!('U', 'T', 1.0, S_matrix[:, 1:k], 0.0, G_matrix)
```

**LAPACK Factorizations**: QR decomposition with pivoting:
```julia
# Use LAPACK for numerically stable QR
LAPACK.geqp3!(A_matrix, jpvt, tau, work)
```

### 3. Algorithmic Optimizations

**Warm Starting**: Initialize with previous solution for related problems.

**Active Set Strategies**: Maintain small active sets through aggressive vertex removal.

**Adaptive Tolerances**: Tighten tolerance as algorithm progresses:
```julia
adaptive_tolerance = max(ε, gap / 1000)
```

### 4. Function Evaluation Caching

**Evaluation Cache**: Store recently computed function values:
```julia
mutable struct CachedSubmodularFunction{F} <: SubmodularFunction
    f::F
    cache::LRU{BitVector, Float64}
    cache_size::Int
end

function evaluate(cf::CachedSubmodularFunction, S::BitVector)
    get!(cf.cache, copy(S)) do
        evaluate(cf.f, S)
    end
end
```

**Marginal Value Caching**: Cache marginal contributions:
```julia
# Cache f(S ∪ {i}) - f(S) for frequently accessed marginals
marginal_cache = Dict{Tuple{BitVector, Int}, Float64}()
```

---

## Memory Management

### Workspace Pre-allocation

**Design Philosophy**: Eliminate allocations in hot paths through comprehensive pre-allocation.

```julia
function create_workspace(n, max_vertices=10*n)
    return WolfeWorkspace(
        n, max_vertices,
        # Pre-allocate all required buffers
        Vector{Int}(undef, n),              # perm_buffer
        falses(n),                          # S_buffer  
        Vector{Float64}(undef, n),          # q_buffer
        Matrix{Float64}(undef, n, max_vertices), # S_matrix
        create_affine_workspace(n, max_vertices),
        # ... all other buffers
    )
end
```

### Memory-Conscious Design

**Buffer Sizing**: Workspaces sized for worst-case scenarios:
```julia
# Conservative sizing for stability
max_vertices = min(max_iterations, 10 * n)
```

**Memory Pooling**: Reuse workspaces across multiple problem instances:
```julia
# Global workspace pool for batch processing
const WORKSPACE_POOL = Vector{WolfeWorkspace}()

function get_workspace(n)
    for ws in WORKSPACE_POOL
        if ws.n == n && !ws.in_use
            ws.in_use = true
            return ws
        end
    end
    # Create new workspace if none available
    push!(WORKSPACE_POOL, WolfeWorkspace(n))
    return WORKSPACE_POOL[end]
end
```

### Garbage Collection Considerations

**Allocation Profiling**: Monitor allocations in critical sections:
```julia
@allocated begin
    # Critical algorithm section
    result = wolfe_algorithm!(workspace, f)
end
```

**Memory Pressure Detection**: Adapt algorithm behavior under memory pressure:
```julia
function adaptive_max_vertices(n, available_memory)
    bytes_per_vertex = sizeof(Float64) * n
    max_by_memory = available_memory ÷ (bytes_per_vertex * 2) # Safety factor
    return min(10 * n, max_by_memory)
end
```

---

## Testing Framework

### Test Categories

#### 1. Correctness Tests
```julia
@testset "Correctness vs Brute Force" begin
    # Compare against exhaustive search for small problems
    for n in 3:8
        f = ConcaveSubmodularFunction(n, 0.7)
        S_bf, val_bf = brute_force_minimization(f)
        S_eff, val_eff, _, _ = fujishige_wolfe_submodular_minimization(f)
        
        @test abs(val_eff - val_bf) < 1e-10
    end
end
```

#### 2. Numerical Stability Tests
```julia
@testset "Numerical Stability" begin
    # Test with ill-conditioned problems
    f = create_ill_conditioned_function(15)
    
    # Test multiple tolerance levels
    for ε in [1e-4, 1e-6, 1e-8]
        S, val, x, iters = fujishige_wolfe_submodular_minimization(f; ε=ε)
        @test isfinite(val)
        @test norm(x) < 1e6  # Reasonable magnitude
    end
end
```

#### 3. Performance Regression Tests
```julia
@testset "Performance Regression" begin
    f = ConcaveSubmodularFunction(20, 0.8)
    
    # Benchmark with known baseline
    time_taken = @elapsed begin
        fujishige_wolfe_submodular_minimization(f)
    end
    
    @test time_taken < 0.1  # Should complete in < 100ms
end
```

#### 4. Memory Tests
```julia
@testset "Memory Management" begin
    workspace = WolfeWorkspace(25)
    f = ConcaveSubmodularFunction(25, 0.6)
    
    # Test pre-allocated version doesn't allocate
    allocs = @allocated begin
        fujishige_wolfe_submodular_minimization!(workspace, f)
    end
    
    @test allocs < 1000  # Minimal allocations
end
```

### Property-Based Testing

**Submodularity Verification**:
```julia
function test_submodularity(f, n_samples=100)
    n = ground_set_size(f)
    for _ in 1:n_samples
        S = rand(Bool, n)
        T = S .| rand(Bool, n)  # T ⊇ S
        v = rand(1:n)
        if !S[v] && !T[v]
            # Test diminishing returns
            margS = marginal_value(f, S, v)
            margT = marginal_value(f, T, v)
            @test margS ≥ margT - 1e-10  # Allow numerical error
        end
    end
end
```

**Algorithm Invariants**:
```julia
function test_algorithm_invariants(workspace, f)
    # Test that current point stays in base polytope
    x = workspace.x_current
    n = length(x)
    
    for subset_size in 1:n
        for S in combinations(1:n, subset_size)
            S_bits = falses(n)
            S_bits[S] .= true
            @test sum(x[S]) ≤ evaluate(f, S_bits) + 1e-10
        end
    end
end
```

---

## Example Functions

### Mathematical Categories

#### 1. Concave Functions
```julia
struct ConcaveSubmodularFunction <: SubmodularFunction
    n::Int
    α::Float64  # 0 < α < 1 for submodularity
end

evaluate(f::ConcaveSubmodularFunction, S::BitVector) = sum(S)^f.α
```

**Applications**: Facility location, coverage problems, diminishing returns modeling.

#### 2. Cut Functions  
```julia
struct CutFunction <: SubmodularFunction
    n::Int
    edges::Vector{Tuple{Int,Int}}
end

function evaluate(f::CutFunction, S::BitVector)
    cut_value = 0
    for (u, v) in f.edges
        if S[u] ⊻ S[v]  # XOR: exactly one endpoint in S
            cut_value += 1
        end
    end
    return cut_value
end
```

**Applications**: Graph partitioning, image segmentation, network analysis.

#### 3. Matroid Rank Functions
```julia
struct MatroidRankFunction <: SubmodularFunction
    n::Int
    k::Int  # Rank constraint
end

evaluate(f::MatroidRankFunction, S::BitVector) = min(sum(S), f.k)
```

**Applications**: Spanning tree problems, linear independence, constraint satisfaction.

### Advanced Functions

#### 4. Log-Determinant Functions
```julia
struct LogDeterminantFunction <: SubmodularFunction
    n::Int
    A::Matrix{Float64}      # Positive semidefinite matrix
    ε::Float64              # Regularization
    log_det_eps_I::Float64  # Normalization constant
end

function evaluate(f::LogDeterminantFunction, S::BitVector)
    active_indices = findall(S)
    if isempty(active_indices)
        return 0.0
    end
    
    A_S = f.A[active_indices, active_indices]
    A_S_reg = A_S + f.ε * I
    return logdet(A_S_reg) - length(active_indices) * log(f.ε)
end
```

**Applications**: Optimal experimental design, sensor placement, information theory.

#### 5. Facility Location Functions
```julia
struct FacilityLocationFunction <: SubmodularFunction
    n::Int  # Number of facilities
    weights::Matrix{Float64}  # weights[customer, facility]
end

function evaluate(f::FacilityLocationFunction, S::BitVector)
    if sum(S) == 0
        return 0.0
    end
    
    total_value = 0.0
    for customer in 1:size(f.weights, 1)
        max_benefit = 0.0
        for facility in 1:f.n
            if S[facility]
                max_benefit = max(max_benefit, f.weights[customer, facility])
            end
        end
        total_value += max_benefit
    end
    return total_value
end
```

**Applications**: Supply chain optimization, resource allocation, service network design.

---

## Advanced Usage

### Custom Submodular Functions

```julia
struct MyCustomFunction <: SubmodularFunction
    n::Int
    parameters::MyParameters
end

ground_set_size(f::MyCustomFunction) = f.n

function evaluate(f::MyCustomFunction, S::BitVector)
    # Implement your submodular function
    # Must satisfy submodularity: f(A) + f(B) ≥ f(A∪B) + f(A∩B)
    return my_computation(S, f.parameters)
end
```

### High-Performance Batch Processing

```julia
function process_batch_efficiently(problems::Vector{SubmodularFunction})
    # Group by size for workspace reuse
    by_size = Dict{Int, Vector{SubmodularFunction}}()
    for prob in problems
        n = ground_set_size(prob)
        get!(by_size, n, SubmodularFunction[])
        push!(by_size[n], prob)
    end
    
    results = []
    for (n, batch) in by_size
        workspace = WolfeWorkspace(n)
        for prob in batch
            result = fujishige_wolfe_submodular_minimization!(workspace, prob)
            push!(results, result)
        end
    end
    
    return results
end
```

### Algorithm Customization

```julia
# Custom convergence criteria
function wolfe_algorithm_custom!(workspace, f; 
                                custom_convergence=default_convergence,
                                max_iterations=10000)
    for iteration in 1:max_iterations
        # ... standard algorithm steps ...
        
        if custom_convergence(workspace, iteration, gap)
            break
        end
    end
    
    return workspace.x_current, iteration, converged
end

# Example: relative gap convergence
function relative_gap_convergence(workspace, iteration, gap)
    x_norm = norm(workspace.x_current)
    relative_gap = abs(gap) / max(x_norm, 1.0)
    return relative_gap < 1e-6
end
```

### Integration with Other Packages

```julia
# JuMP.jl integration for mixed-integer extensions
using JuMP, GLPK

function solve_constrained_submodular_minimization(f, cardinality_bound)
    model = Model(GLPK.Optimizer)
    n = ground_set_size(f)
    
    @variable(model, x[1:n], Bin)
    @constraint(model, sum(x) ≤ cardinality_bound)
    
    # Use Wolfe algorithm for continuous relaxation bound
    S_min, val_min, _, _ = fujishige_wolfe_submodular_minimization(f)
    @objective(model, Min, sum(x[i] * marginal_value(f, S_min, i) for i in 1:n))
    
    optimize!(model)
    return value.(x), objective_value(model)
end
```

---

## References

1. **Chakrabarty, Jain, Kothari (2014)**: "Provable Submodular Minimization using Wolfe's Algorithm"
2. **Fujishige (1980)**: "Lexicographically optimal base of a polyhedron with respect to a weight vector"
3. **Wolfe (1976)**: "Finding the nearest point in a polytope"
4. **Frank, Wolfe (1956)**: "An algorithm for quadratic programming"
5. **Lovász (1983)**: "Submodular functions and convexity"

This implementation represents the state-of-the-art in submodular minimization, combining theoretical rigor with practical performance optimizations for real-world applications.