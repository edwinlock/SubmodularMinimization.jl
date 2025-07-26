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
11. [Python Wrapper Architecture](#python-wrapper-architecture)
12. [Advanced Usage](#advanced-usage)

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

**Selective Caching Strategy**: Caching is disabled by default in Julia algorithms but enabled for specific use cases:

```julia
# Default behavior (no caching overhead)
fujishige_wolfe_submodular_minimization(f; cache=false)  # Julia default

# Enable caching for expensive functions or Python interop
fujishige_wolfe_submodular_minimization(f; cache=true)   # Explicit enable

# Always cache for exponential-complexity functions
is_submodular(f; cache=true)  # O(n² × 2ⁿ) benefits from caching
```

**Caching Infrastructure**:
```julia
mutable struct CachedSubmodularFunction{F} <: SubmodularFunction
    f::F
    cache::Dict{Vector{Int}, Float64}
    max_cache_size::Int
    stats::CacheStats
end

function evaluate(cf::CachedSubmodularFunction, S::BitVector)
    key = findall(S)  # Convert to indices for hashing
    
    if haskey(cf.cache, key)
        cf.stats.hits += 1
        return cf.cache[key]
    end
    
    cf.stats.misses += 1
    result = evaluate(cf.f, S)
    
    # Bounded cache with LRU eviction
    if length(cf.cache) >= cf.max_cache_size && cf.max_cache_size > 0
        evict_lru_entry!(cf.cache)
    end
    
    cf.cache[copy(key)] = result
    return result
end
```

**Performance Considerations**:
- **Main algorithms**: Caching disabled by default (adds overhead without benefit)
- **Verification functions**: Caching enabled by default (high redundancy in O(2ⁿ) checks)
- **Python wrapper**: Caching enabled by default (reduces interop overhead)
- **Custom thresholds**: Auto-recommendations based on problem characteristics

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

## Verification and Testing Tools

### Submodularity Verification: `is_submodular()`

The `is_submodular()` function implements Definition 3 of submodularity to verify whether a given function is actually submodular:

```julia
function is_submodular(f::SubmodularFunction; tolerance::Float64=COMPARISON_TOLERANCE, verbose::Bool=false)
    n = ground_set_size(f)
    violations = 0
    total_tests = 0
    
    # Test all possible combinations using Definition 3:
    # For all X ⊆ V and x1, x2 ∉ X with x1 ≠ x2:
    # f(X ∪ {x1}) + f(X ∪ {x2}) ≥ f(X ∪ {x1, x2}) + f(X)
    
    for x_bits in 0:(2^n - 1)
        X = BitVector([((x_bits >> (i-1)) & 1) == 1 for i in 1:n])
        
        for x1 in 1:n, x2 in (x1+1):n
            if !X[x1] && !X[x2]  # Both elements not in X
                X_x1 = copy(X); X_x1[x1] = true
                X_x2 = copy(X); X_x2[x2] = true  
                X_x1_x2 = copy(X); X_x1_x2[x1] = true; X_x1_x2[x2] = true
                
                lhs = evaluate(f, X_x1) + evaluate(f, X_x2)
                rhs = evaluate(f, X_x1_x2) + evaluate(f, X)
                
                total_tests += 1
                if lhs < rhs - tolerance
                    violations += 1
                    if verbose
                        println("Violation: X=$(findall(X)), x1=$x1, x2=$x2")
                        println("  LHS=$lhs, RHS=$rhs, diff=$(rhs-lhs)")
                    end
                end
            end
        end
    end
    
    return violations == 0, violations, total_tests
end
```

**Complexity**: O(n² × 2ⁿ) - only practical for n ≤ 10.

### Optimality Verification: `is_minimiser()`

The `is_minimiser()` function checks whether a given solution is optimal for a submodular function using local optimality conditions:

```julia
function is_minimiser(S::BitVector, f::SubmodularFunction; tolerance::Float64=COMPARISON_TOLERANCE, verbose::Bool=false)
    n = ground_set_size(f)
    current_value = evaluate(f, S)
    
    # For submodular functions: local optimality ⟹ global optimality
    # Check all single-element additions and removals
    
    for i in 1:n
        if S[i]  # Element i is in the set - try removing it
            S_minus = copy(S)
            S_minus[i] = false
            new_value = evaluate(f, S_minus)
            
            if new_value < current_value - tolerance
                return false, "removing element $i", new_value
            end
        else  # Element i is not in the set - try adding it
            S_plus = copy(S)
            S_plus[i] = true
            new_value = evaluate(f, S_plus)
            
            if new_value < current_value - tolerance
                return false, "adding element $i", new_value
            end
        end
    end
    
    return true, "", NaN
end
```

**Key Properties**:
- **Complexity**: O(n) - scales linearly with problem size
- **Theoretical Foundation**: For submodular functions, local optimality implies global optimality
- **Practical**: Works for any problem size, unlike brute force verification
- **Informative**: Provides specific improvement suggestions when solution is suboptimal

### Testing Framework

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

## Python Wrapper Architecture

### Overview

The Python wrapper provides access to SubmodularMinimization.jl's high-performance algorithms through a C-compatible dynamic library created with PackageCompiler.jl. This approach combines Julia's computational performance with Python's ease of use.

### Architecture Components

```
Python Layer
    ↓ ctypes
C Interface (@ccallable functions)
    ↓ Julia FFI
Julia Implementation
    ↓ BLAS/LAPACK
Native Libraries
```

### Build System: PackageCompiler.jl

**Compilation Process**:
```julia
# build_python_library.jl
using PackageCompiler

create_library(
    ".",  # Current package
    "python_lib";
    lib_name = "libsubmodular",
    precompile_execution_file = "build_precompile.jl",
    force = true
)
```

**Precompilation Strategy**:
```julia
# Precompile common function types and sizes
for n in [5, 10, 20, 50]
    for alpha in [0.3, 0.5, 0.7, 0.9]
        f = ConcaveSubmodularFunction(n, alpha)
        fujishige_wolfe_submodular_minimization(f; ε=1e-4, verbose=false)
        is_submodular(f; verbose=false)
        is_minimiser(result.optimal_set, f; verbose=false)
    end
end
```

**Output Structure**:
```
python_lib/
├── lib/
│   ├── libsubmodular.so     # Linux shared library
│   ├── libsubmodular.dylib  # macOS dynamic library  
│   └── libsubmodular.dll    # Windows DLL
└── include/
    └── submodular_minimization.h  # C header file
```

### C Interface Layer

**Function Signatures**:
```c
// Main algorithms
int32_t fujishige_wolfe_solve_c(
    int32_t func_type, double* params, int32_t n_params, int32_t n,
    double tolerance, int32_t max_iterations,
    int32_t** result_set, double* result_value, int32_t* result_iterations
);

int32_t wolfe_algorithm_c(
    int32_t func_type, double* params, int32_t n_params, int32_t n,
    double tolerance, int32_t max_iterations,
    double** result_x, int32_t* result_iterations, int32_t* converged
);

// Verification functions
int32_t check_submodular_c(
    int32_t func_type, double* params, int32_t n_params, int32_t n,
    int32_t* violations
);

int32_t is_minimiser_c(
    int32_t func_type, double* params, int32_t n_params, int32_t n,
    int32_t* candidate_set, int32_t set_size, double* improvement_value
);
```

**Memory Management**:
```julia
# Julia allocates result arrays and returns pointers
result_array = [Cint(idx - 1) for idx in selected_indices]  # 0-indexed for C
unsafe_store!(result_set, pointer(result_array))

# Python responsible for not accessing freed memory
# GC handles deallocation automatically
```

**Error Handling**:
```julia
const SUCCESS = Cint(0)
const ERROR_INVALID_FUNCTION_TYPE = Cint(-1)
const ERROR_INVALID_PARAMETERS = Cint(-2)
const ERROR_CONVERGENCE_FAILED = Cint(-3)
const ERROR_MEMORY_ALLOCATION = Cint(-4)
```

### Python Object Model

**Function Type Hierarchy**:
```python
class SubmodularFunction(ABC):
    @property
    @abstractmethod 
    def func_type(self) -> int: pass
    
    @property
    @abstractmethod
    def parameters(self) -> List[float]: pass

class ConcaveFunction(SubmodularFunction):
    def __init__(self, n: int, alpha: float):
        self.alpha = alpha
    
    @property
    def func_type(self) -> int:
        return 1  # FUNC_TYPE_CONCAVE
    
    @property 
    def parameters(self) -> List[float]:
        return [self.alpha]

class CallbackFunction(SubmodularFunction):
    def __init__(self, callback: Callable[[List[int]], float], n: int):
        self.callback = callback
        self._callback_wrapper = None
    
    def get_callback_wrapper(self):
        CALLBACK_TYPE = ctypes.CFUNCTYPE(
            ctypes.c_double, 
            ctypes.POINTER(ctypes.c_int32), 
            ctypes.c_int32
        )
        
        def c_callback(indices_ptr, n_indices):
            indices = [indices_ptr[i] for i in range(n_indices)]
            return float(self.callback(indices))
        
        return CALLBACK_TYPE(c_callback)
```

**Result Types**:
```python
class SubmodularResult(NamedTuple):
    optimal_set: List[int]
    min_value: float
    iterations: int
    success: bool
    error_message: str = ""

class WolfeResult(NamedTuple):
    min_norm_point: List[float]
    iterations: int
    converged: bool
    success: bool
    error_message: str = ""
```

### Memory Management and Performance

**Index Convention Translation**:
```python
# Python uses 0-indexed, Julia uses 1-indexed
# Conversion happens at C interface boundary

# Python → Julia
julia_indices = [idx + 1 for idx in python_indices]

# Julia → Python  
python_indices = [idx - 1 for idx in julia_indices]
```

**Caching Strategy**:
```julia
# Python wrapper enables caching by default
fujishige_wolfe_submodular_minimization(f; cache=true)  # Python calls
fujishige_wolfe_submodular_minimization(f; cache=false) # Julia calls

# Callback-specific memoization
const MEMOIZATION_CACHE = Dict{Vector{Int}, Float64}()

function evaluate(f::CallbackSubmodularFunction, S::BitVector)
    indices = [i - 1 for i in findall(S)]
    
    if haskey(MEMOIZATION_CACHE, indices)
        CACHE_HITS[] += 1
        return MEMOIZATION_CACHE[indices]
    end
    
    CACHE_MISSES[] += 1
    result = ccall(f.callback_ptr, Cdouble, (Ptr{Cint}, Cint), 
                   indices_array, Cint(length(indices)))
    
    MEMOIZATION_CACHE[copy(indices)] = Float64(result)
    return Float64(result)
end
```

**Performance Optimizations**:
- **Zero-copy data sharing** where possible via pointer passing
- **Batch parameter marshaling** to minimize FFI overhead
- **Automatic caching** for callback functions to reduce Python↔Julia transitions
- **Pre-allocated result buffers** managed by Julia GC

### Library Loading and Discovery

**Automatic Library Detection**:
```python
def _load_library(self, library_path: Optional[str]):
    if library_path is None:
        candidates = [
            "python_lib/lib/libsubmodular.so",    # Build output
            "python_lib/lib/libsubmodular.dylib", # macOS
            "python_lib/lib/libsubmodular.dll",   # Windows
            "libsubmodular.so",                   # System path
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                library_path = candidate
                break
        else:
            raise FileNotFoundError("Library not found. Run: julia build_python_library.jl")
    
    self.lib = ctypes.CDLL(library_path)
```

**Environment Setup**:
```bash
# Required environment variables
export LD_LIBRARY_PATH=$PWD/python_lib/lib:$LD_LIBRARY_PATH      # Linux
export DYLD_LIBRARY_PATH=$PWD/python_lib/lib:$DYLD_LIBRARY_PATH  # macOS  
export PATH=$PWD/python_lib/lib:$PATH                            # Windows
```

### Error Handling and Diagnostics

**Comprehensive Error Messages**:
```python
def solve(self, func: SubmodularFunction, tolerance: float = 1e-6, 
          max_iterations: int = 10000) -> SubmodularResult:
    
    set_size = self.lib.fujishige_wolfe_solve_c(...)
    
    if set_size < 0:
        error_messages = {
            self.ERROR_INVALID_FUNCTION_TYPE: "Invalid function type",
            self.ERROR_INVALID_PARAMETERS: "Invalid parameters",
            self.ERROR_CONVERGENCE_FAILED: "Algorithm convergence failed",
            self.ERROR_MEMORY_ALLOCATION: "Memory allocation error"
        }
        return SubmodularResult(
            optimal_set=[], min_value=float('nan'), iterations=0, 
            success=False, 
            error_message=error_messages.get(set_size, f"Unknown error: {set_size}")
        )
```

**Cache Performance Monitoring**:
```python
def get_cache_stats(self):
    hits = ctypes.c_int32()
    misses = ctypes.c_int32() 
    size = ctypes.c_int32()
    
    self.lib.get_cache_stats(ctypes.byref(hits), ctypes.byref(misses), ctypes.byref(size))
    
    total_calls = hits.value + misses.value
    hit_rate = hits.value / total_calls if total_calls > 0 else 0.0
    
    return {
        'cache_hits': hits.value,
        'cache_misses': misses.value, 
        'cache_size': size.value,
        'hit_rate': hit_rate
    }
```

### Testing Infrastructure

**Multi-Language Test Suite**:
```python
# test_python_wrapper.py
def test_fujishige_wolfe():
    solver = SubmodularMinimizer()
    
    # Test concave function
    result = solver.solve_concave(n=8, alpha=0.7, tolerance=1e-6)
    assert result.success
    assert result.min_value >= 0
    assert len(result.optimal_set) <= 8
    
    # Test cut function
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    result = solver.solve_cut(n=4, edges=edges)
    assert result.success
    
    # Test callback function
    def custom_func(indices):
        return len(indices) ** 0.6
    
    callback_func = solver.create_callback_function(custom_func, n=6)
    result = solver.solve(callback_func)
    assert result.success
```

**Performance Validation**:
```python
def benchmark_python_vs_julia():
    # Compare Python wrapper performance to pure Julia
    import time
    
    solver = SubmodularMinimizer()
    func = solver.create_concave_function(n=20, alpha=0.7)
    
    # Python wrapper timing
    start = time.time()
    result = solver.solve(func)
    python_time = time.time() - start
    
    # Verify performance is within 2x of Julia native
    assert python_time < expected_julia_time * 2.0
```

### Integration Patterns

**Jupyter Notebook Usage**:
```python
import numpy as np
import matplotlib.pyplot as plt
from submodular_minimization_python import SubmodularMinimizer

solver = SubmodularMinimizer()

# Interactive exploration
for alpha in np.linspace(0.1, 0.9, 9):
    func = solver.create_concave_function(n=10, alpha=alpha)
    result = solver.solve(func)
    plt.scatter(alpha, len(result.optimal_set))

plt.xlabel('Alpha parameter')  
plt.ylabel('Optimal set size')
plt.title('Concave Function Optimization')
plt.show()
```

**Scikit-learn Integration**:
```python
from sklearn.base import BaseEstimator
from submodular_minimization_python import SubmodularMinimizer

class SubmodularFeatureSelector(BaseEstimator):
    def __init__(self, alpha=0.7, max_features=None):
        self.alpha = alpha
        self.max_features = max_features
        self.solver = SubmodularMinimizer()
    
    def fit(self, X, y):
        def feature_utility(feature_indices):
            # Submodular utility based on feature correlation
            return self._compute_submodular_utility(X[:, feature_indices], y)
        
        n_features = X.shape[1]
        func = self.solver.create_callback_function(feature_utility, n_features)
        result = self.solver.solve(func)
        
        self.selected_features_ = result.optimal_set
        return self
```

This architecture provides a robust, high-performance bridge between Julia's computational capabilities and Python's ecosystem, maintaining the performance benefits of the original Julia implementation while offering the familiar Python interface that many data scientists and researchers prefer.

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