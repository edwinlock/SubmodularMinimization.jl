"""
Example Submodular Function Constructors

This module contains concrete implementations of common submodular functions
for testing and demonstration purposes.
"""

"""
    ConcaveSubmodularFunction

Example submodular function: f(S) = |S|^α where 0 < α < 1 (submodular concave function)
"""
struct ConcaveSubmodularFunction <: SubmodularFunction
    n::Int
    α::Float64
    
    function ConcaveSubmodularFunction(n::Int, α::Float64)
        n > 0 || throw(ArgumentError("Ground set size n must be positive, got $n"))
        0 < α < 1 || throw(ArgumentError("α must be between 0 and 1 for submodularity, got $α"))
        new(n, α)
    end
end

# Convenience constructor with default α
ConcaveSubmodularFunction(n::Int) = ConcaveSubmodularFunction(n, 0.7)

ground_set_size(f::ConcaveSubmodularFunction) = f.n
function evaluate(f::ConcaveSubmodularFunction, S::BitVector) 
    length(S) == f.n || throw(ArgumentError("BitVector length ($(length(S))) must match ground set size ($(f.n))"))
    return sum(S)^f.α
end

"""
    CutFunction

Graph cut function: f(S) = number of edges between S and complement of S
"""
struct CutFunction <: SubmodularFunction
    n::Int
    edges::Vector{Tuple{Int,Int}}
    
    function CutFunction(n::Int, edges::Vector{Tuple{Int,Int}})
        n > 0 || throw(ArgumentError("Ground set size n must be positive, got $n"))
        
        # Validate edges
        for (i, (u, v)) in enumerate(edges)
            1 <= u <= n || throw(BoundsError("Edge $i: vertex $u is out of range [1, $n]"))
            1 <= v <= n || throw(BoundsError("Edge $i: vertex $v is out of range [1, $n]"))
            u != v || throw(ArgumentError("Edge $i: self-loops not allowed ($u, $v)"))
            u < v || throw(ArgumentError("Edge $i: edges should be ordered as (u, v) with u < v, got ($u, $v)"))
        end
        
        new(n, edges)
    end
end

ground_set_size(f::CutFunction) = f.n

function evaluate(f::CutFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length ($(length(S))) must match ground set size ($(f.n))"))
    
    cut_value = 0
    for (u, v) in f.edges
        if S[u] ⊻ S[v]  # XOR: true if exactly one endpoint is in S
            cut_value += 1
        end
    end
    return cut_value
end

"""
    create_random_cut_function(n::Int, p::Float64)

Create a random cut function on n vertices with edge probability p.
Returns a CutFunction representing the cut function of a random graph.
"""
function create_random_cut_function(n::Int, p::Float64)
    # Input validation
    n > 0 || throw(ArgumentError("Ground set size n must be positive, got $n"))
    0 <= p <= 1 || throw(ArgumentError("Edge probability p must be in [0, 1], got $p"))
    
    edges = Tuple{Int,Int}[]
    for i in 1:n
        for j in i+1:n
            if rand() < p
                push!(edges, (i, j))
            end
        end
    end
    return CutFunction(n, edges)
end

"""
    MatroidRankFunction

Rank function of a uniform matroid: f(S) = min(|S|, k)
Represents selecting at most k elements from n, normalized so f(∅) = 0.
"""
struct MatroidRankFunction <: SubmodularFunction
    n::Int
    k::Int
    
    function MatroidRankFunction(n::Int, k::Int)
        n > 0 || throw(ArgumentError("Ground set size n must be positive, got $n"))
        0 <= k <= n || throw(ArgumentError("Rank k must be in [0, $n], got $k"))
        new(n, k)
    end
end

ground_set_size(f::MatroidRankFunction) = f.n

function evaluate(f::MatroidRankFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length ($(length(S))) must match ground set size ($(f.n))"))
    return min(sum(S), f.k)
end

"""
    FacilityLocationFunction

Facility location function: f(S) = Σᵢ max_{j∈S} wᵢⱼ
where wᵢⱼ is the benefit of serving customer i from facility j.
Normalized so that f(∅) = 0 (no facilities serve any customers).
"""
struct FacilityLocationFunction <: SubmodularFunction
    n::Int  # number of facilities
    weights::Matrix{Float64}  # weights[customer, facility]
    
    function FacilityLocationFunction(n::Int, weights::Matrix{Float64})
        n > 0 || throw(ArgumentError("Number of facilities n must be positive, got $n"))
        size(weights, 2) == n || throw(ArgumentError("Weight matrix must have $n columns (facilities), got $(size(weights, 2))"))
        size(weights, 1) > 0 || throw(ArgumentError("Weight matrix must have at least one row (customer)"))
        all(w >= 0 for w in weights) || throw(ArgumentError("All weights must be non-negative"))
        new(n, weights)
    end
end

ground_set_size(f::FacilityLocationFunction) = f.n

function evaluate(f::FacilityLocationFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length ($(length(S))) must match ground set size ($(f.n))"))
    
    if sum(S) == 0  # Empty set
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

"""
    LogDeterminantFunction

Log-determinant function: f(S) = log det(A_S + εI) - log det(εI)
where A_S is the submatrix of A indexed by S, and ε is a regularization parameter.
This is normalized so that f(∅) = 0.
"""
struct LogDeterminantFunction <: SubmodularFunction
    n::Int
    A::Matrix{Float64}
    ε::Float64
    log_det_eps_I::Float64  # Pre-computed log det(εI) for normalization
    
    function LogDeterminantFunction(n::Int, A::Matrix{Float64}, ε::Float64=1e-6)
        n > 0 || throw(ArgumentError("Ground set size n must be positive, got $n"))
        size(A) == (n, n) || throw(ArgumentError("Matrix A must be $n×$n, got $(size(A))"))
        ε > 0 || throw(ArgumentError("Regularization parameter ε must be positive, got $ε"))
        
        # Check if A is positive semidefinite (approximately)
        eigenvals = eigvals(A)
        all(λ >= -1e-10 for λ in eigenvals) || @warn "Matrix A may not be positive semidefinite"
        
        # Pre-compute normalization constant
        log_det_eps_I = n * log(ε)
        
        new(n, A, ε, log_det_eps_I)
    end
end

ground_set_size(f::LogDeterminantFunction) = f.n

function evaluate(f::LogDeterminantFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length ($(length(S))) must match ground set size ($(f.n))"))
    
    active_indices = findall(S)
    k = length(active_indices)
    
    if k == 0  # Empty set
        return 0.0
    end
    
    # Extract submatrix A_S
    A_S = f.A[active_indices, active_indices]
    
    # Add regularization: A_S + εI
    A_S_reg = A_S + f.ε * I
    
    # Compute log determinant and normalize
    return logdet(A_S_reg) - k * log(f.ε)
end

"""
    EntropyFunction

Entropy-based submodular function: f(S) = H(X_S) where H is entropy.
For discrete distributions, this represents the entropy of variables indexed by S.
Normalized so that f(∅) = 0.
"""
struct EntropyFunction <: SubmodularFunction
    n::Int
    probabilities::Matrix{Float64}  # probabilities[outcome, variable]
    
    function EntropyFunction(n::Int, probabilities::Matrix{Float64})
        n > 0 || throw(ArgumentError("Ground set size n must be positive, got $n"))
        size(probabilities, 2) == n || throw(ArgumentError("Probability matrix must have $n columns (variables), got $(size(probabilities, 2))"))
        size(probabilities, 1) > 0 || throw(ArgumentError("Probability matrix must have at least one row (outcome)"))
        
        # Validate probabilities
        for j in 1:n
            col_sum = sum(probabilities[:, j])
            abs(col_sum - 1.0) < 1e-10 || throw(ArgumentError("Column $j probabilities must sum to 1, got $col_sum"))
            all(p >= 0 for p in probabilities[:, j]) || throw(ArgumentError("All probabilities must be non-negative"))
        end
        
        new(n, probabilities)
    end
end

ground_set_size(f::EntropyFunction) = f.n

function evaluate(f::EntropyFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length ($(length(S))) must match ground set size ($(f.n))"))
    
    active_indices = findall(S)
    k = length(active_indices)
    
    if k == 0  # Empty set
        return 0.0
    end
    
    # Compute joint entropy H(X_S)
    # This is a simplified version - in practice, would need joint distribution
    # Here we use the sum of marginal entropies as an approximation (upper bound)
    total_entropy = 0.0
    for j in active_indices
        marginal_entropy = 0.0
        for i in 1:size(f.probabilities, 1)
            p = f.probabilities[i, j]
            if p > 0
                marginal_entropy -= p * log(p)
            end
        end
        total_entropy += marginal_entropy
    end
    
    return total_entropy
end

"""
    SquareRootFunction

Square root submodular function: f(S) = √|S|
Simple concave function that's submodular, normalized so f(∅) = 0.
"""
struct SquareRootFunction <: SubmodularFunction
    n::Int
    
    function SquareRootFunction(n::Int)
        n > 0 || throw(ArgumentError("Ground set size n must be positive, got $n"))
        new(n)
    end
end

ground_set_size(f::SquareRootFunction) = f.n

function evaluate(f::SquareRootFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length ($(length(S))) must match ground set size ($(f.n))"))
    return sqrt(sum(S))
end

"""
    WeightedCoverageFunction

Weighted coverage function: f(S) = total weight of elements covered by sets in S.
Each element i has weight wᵢ, and each set j covers a subset of elements.
Normalized so that f(∅) = 0 (no sets cover anything).
"""
struct WeightedCoverageFunction <: SubmodularFunction
    n::Int  # number of sets
    element_weights::Vector{Float64}
    coverage_matrix::BitMatrix  # coverage_matrix[element, set] = true if set covers element
    
    function WeightedCoverageFunction(n::Int, element_weights::Vector{Float64}, coverage_matrix::BitMatrix)
        n > 0 || throw(ArgumentError("Number of sets n must be positive, got $n"))
        size(coverage_matrix, 2) == n || throw(ArgumentError("Coverage matrix must have $n columns (sets), got $(size(coverage_matrix, 2))"))
        length(element_weights) == size(coverage_matrix, 1) || throw(ArgumentError("Element weights length ($(length(element_weights))) must match number of elements ($(size(coverage_matrix, 1)))"))
        all(w >= 0 for w in element_weights) || throw(ArgumentError("All element weights must be non-negative"))
        
        new(n, element_weights, coverage_matrix)
    end
end

ground_set_size(f::WeightedCoverageFunction) = f.n

function evaluate(f::WeightedCoverageFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length ($(length(S))) must match ground set size ($(f.n))"))
    
    if sum(S) == 0  # Empty set
        return 0.0
    end
    
    total_weight = 0.0
    for element in 1:size(f.coverage_matrix, 1)
        # Check if this element is covered by any selected set
        covered = false
        for set in 1:f.n
            if S[set] && f.coverage_matrix[element, set]
                covered = true
                break
            end
        end
        if covered
            total_weight += f.element_weights[element]
        end
    end
    
    return total_weight
end

"""
    BipartiteMatchingFunction

Represents the negative of a maximum bipartite matching objective.
The minimizer will be a subset that induces the largest possible matching.
"""
struct BipartiteMatchingFunction <: SubmodularFunction
    n::Int  # total number of vertices (|U| + |V|)
    left_size::Int  # |U|
    edges::Vector{Tuple{Int,Int}}  # edges between left and right vertices
    
    function BipartiteMatchingFunction(left_size::Int, right_size::Int, edges::Vector{Tuple{Int,Int}})
        left_size > 0 || throw(ArgumentError("Left partition size must be positive"))
        right_size > 0 || throw(ArgumentError("Right partition size must be positive"))
        n = left_size + right_size
        
        # Validate edges
        for (u, v) in edges
            1 <= u <= left_size || throw(ArgumentError("Left vertex $u must be in range [1, $left_size]"))
            (left_size + 1) <= v <= n || throw(ArgumentError("Right vertex $v must be in range [$(left_size + 1), $n]"))
        end
        
        new(n, left_size, edges)
    end
end

ground_set_size(f::BipartiteMatchingFunction) = f.n

function evaluate(f::BipartiteMatchingFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    # Count edges where both endpoints are selected
    matching_edges = 0
    for (u, v) in f.edges
        if S[u] && S[v]
            matching_edges += 1
        end
    end
    
    # Return negative matching size (since we want to maximize matching via minimization)
    return -Float64(matching_edges)
end

"""
    AsymmetricCutFunction

Cut function with asymmetric penalties - cutting edges in one direction costs more.
"""
struct AsymmetricCutFunction <: SubmodularFunction
    n::Int
    edges::Vector{Tuple{Int,Int}}
    forward_weights::Vector{Float64}  # cost of cutting edge (u,v) when u ∈ S, v ∉ S
    backward_weights::Vector{Float64}  # cost of cutting edge (u,v) when u ∉ S, v ∈ S
    
    function AsymmetricCutFunction(n::Int, edges::Vector{Tuple{Int,Int}}, 
                                 forward_weights::Vector{Float64}, backward_weights::Vector{Float64})
        n > 0 || throw(ArgumentError("Ground set size must be positive"))
        length(edges) == length(forward_weights) == length(backward_weights) || 
            throw(ArgumentError("Edges and weights must have same length"))
        all(w >= 0 for w in forward_weights) || throw(ArgumentError("Forward weights must be non-negative"))
        all(w >= 0 for w in backward_weights) || throw(ArgumentError("Backward weights must be non-negative"))
        
        for (u, v) in edges
            1 <= u <= n && 1 <= v <= n || throw(ArgumentError("Edge endpoints must be in [1, $n]"))
            u != v || throw(ArgumentError("Self-loops not allowed"))
        end
        
        new(n, edges, forward_weights, backward_weights)
    end
end

ground_set_size(f::AsymmetricCutFunction) = f.n

function evaluate(f::AsymmetricCutFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    total_cost = 0.0
    for (i, (u, v)) in enumerate(f.edges)
        if S[u] && !S[v]
            # Forward cut: u ∈ S, v ∉ S
            total_cost += f.forward_weights[i]
        elseif !S[u] && S[v]
            # Backward cut: u ∉ S, v ∈ S  
            total_cost += f.backward_weights[i]
        end
    end
    
    return total_cost
end

"""
    ConcaveWithPenalty

Concave function with a penalty for empty and full sets to force non-trivial minimizers.
"""
struct ConcaveWithPenalty <: SubmodularFunction
    n::Int
    α::Float64
    penalty::Float64
    
    function ConcaveWithPenalty(n::Int, α::Float64, penalty::Float64)
        n > 0 || throw(ArgumentError("Ground set size must be positive"))
        0 < α < 1 || throw(ArgumentError("Concavity parameter α must be in (0,1)"))
        penalty > 0 || throw(ArgumentError("Penalty must be positive"))
        new(n, α, penalty)
    end
end

ground_set_size(f::ConcaveWithPenalty) = f.n

function evaluate(f::ConcaveWithPenalty, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    cardinality = sum(S)
    base_value = Float64(cardinality)^f.α
    
    # Add penalty for empty or full set
    if cardinality == 0 || cardinality == f.n
        base_value += f.penalty
    end
    
    return base_value
end

# ============================================================================
# Helper constructor functions for common use cases
# ============================================================================

"""
    create_random_facility_location(n_facilities::Int, n_customers::Int; max_weight::Float64=10.0)

Create a random facility location function with given number of facilities and customers.
"""
function create_random_facility_location(n_facilities::Int, n_customers::Int; max_weight::Float64=10.0)
    n_facilities > 0 || throw(ArgumentError("Number of facilities must be positive, got $n_facilities"))
    n_customers > 0 || throw(ArgumentError("Number of customers must be positive, got $n_customers"))
    max_weight > 0 || throw(ArgumentError("Maximum weight must be positive, got $max_weight"))
    
    # Generate random weights with some structure (closer facilities have higher weight)
    weights = rand(n_customers, n_facilities) * max_weight
    return FacilityLocationFunction(n_facilities, weights)
end

"""
    create_random_coverage_function(n_sets::Int, n_elements::Int; coverage_prob::Float64=0.3, max_weight::Float64=5.0)

Create a random weighted coverage function.
"""
function create_random_coverage_function(n_sets::Int, n_elements::Int; coverage_prob::Float64=0.3, max_weight::Float64=5.0)
    n_sets > 0 || throw(ArgumentError("Number of sets must be positive, got $n_sets"))
    n_elements > 0 || throw(ArgumentError("Number of elements must be positive, got $n_elements"))
    0 <= coverage_prob <= 1 || throw(ArgumentError("Coverage probability must be in [0,1], got $coverage_prob"))
    max_weight > 0 || throw(ArgumentError("Maximum weight must be positive, got $max_weight"))
    
    # Random element weights
    element_weights = rand(n_elements) * max_weight
    
    # Random coverage matrix
    coverage_matrix = rand(n_elements, n_sets) .< coverage_prob
    
    return WeightedCoverageFunction(n_sets, element_weights, coverage_matrix)
end

"""
    create_wishart_log_determinant(n::Int, degrees_freedom::Int=n+5)

Create a log-determinant function using a random Wishart matrix (guaranteed positive definite).
"""
function create_wishart_log_determinant(n::Int, degrees_freedom::Int=n+5)
    n > 0 || throw(ArgumentError("Ground set size must be positive, got $n"))
    degrees_freedom >= n || throw(ArgumentError("Degrees of freedom must be at least $n, got $degrees_freedom"))
    
    # Generate a random positive definite matrix using Wishart distribution
    # A = X'X where X is degrees_freedom × n random matrix
    X = randn(degrees_freedom, n)
    A = X' * X / degrees_freedom  # Normalize by degrees of freedom
    
    return LogDeterminantFunction(n, A)
end

"""
    create_random_entropy_function(n::Int, n_outcomes::Int=4)

Create a random entropy function with specified number of variables and outcomes.
"""
function create_random_entropy_function(n::Int, n_outcomes::Int=4)
    n > 0 || throw(ArgumentError("Number of variables must be positive, got $n"))
    n_outcomes > 0 || throw(ArgumentError("Number of outcomes must be positive, got $n_outcomes"))
    
    # Generate random probability distributions for each variable
    probabilities = zeros(n_outcomes, n)
    for j in 1:n
        # Generate random probabilities and normalize
        raw_probs = rand(n_outcomes)
        probabilities[:, j] = raw_probs / sum(raw_probs)
    end
    
    return EntropyFunction(n, probabilities)
end

"""
    create_bipartite_matching(left_size::Int, right_size::Int, edge_prob::Float64=0.3)

Create a random bipartite matching function.
"""
function create_bipartite_matching(left_size::Int, right_size::Int, edge_prob::Float64=0.3)
    edges = Tuple{Int,Int}[]
    for u in 1:left_size
        for v in (left_size+1):(left_size+right_size)
            if rand() < edge_prob
                push!(edges, (u, v))
            end
        end
    end
    return BipartiteMatchingFunction(left_size, right_size, edges)
end

"""
    create_asymmetric_cut(n::Int, edge_prob::Float64=0.4, asymmetry_factor::Float64=3.0)

Create an asymmetric cut function where forward and backward edge weights differ.
"""
function create_asymmetric_cut(n::Int, edge_prob::Float64=0.4, asymmetry_factor::Float64=3.0)
    edges = Tuple{Int,Int}[]
    forward_weights = Float64[]
    backward_weights = Float64[]
    
    for i in 1:n, j in i+1:n
        if rand() < edge_prob
            push!(edges, (i, j))
            base_weight = rand() + 0.1  # Avoid zero weights
            push!(forward_weights, base_weight)
            push!(backward_weights, base_weight * asymmetry_factor)
        end
    end
    
    return AsymmetricCutFunction(n, edges, forward_weights, backward_weights)
end

"""
    create_concave_with_penalty(n::Int, α::Float64=0.5, penalty::Float64=2.0)

Create a concave function with penalty for trivial solutions.
"""
function create_concave_with_penalty(n::Int, α::Float64=0.5, penalty::Float64=2.0)
    return ConcaveWithPenalty(n, α, penalty)
end

# ============================================================================
# AI and Image Recognition Submodular Functions
# ============================================================================

"""
    ImageSegmentationFunction

Graph-based image segmentation function using edge weights and region consistency.
Commonly used in computer vision for semantic segmentation tasks.

This function models the cost of segmenting an image into foreground/background regions,
where the cost includes boundary length (edge cuts) and region consistency terms.
"""
struct ImageSegmentationFunction <: SubmodularFunction
    n::Int  # number of pixels/superpixels
    edge_weights::Matrix{Float64}  # pairwise similarity weights
    unary_costs::Vector{Float64}   # per-pixel classification costs
    λ::Float64  # balance between unary and pairwise terms
    
    function ImageSegmentationFunction(n::Int, edge_weights::Matrix{Float64}, 
                                     unary_costs::Vector{Float64}, λ::Float64=1.0)
        n > 0 || throw(ArgumentError("Number of pixels must be positive, got $n"))
        size(edge_weights) == (n, n) || throw(ArgumentError("Edge weights must be $n×$n, got $(size(edge_weights))"))
        length(unary_costs) == n || throw(ArgumentError("Unary costs length must be $n, got $(length(unary_costs))"))
        λ >= 0 || throw(ArgumentError("Lambda must be non-negative, got $λ"))
        all(w >= 0 for w in edge_weights) || throw(ArgumentError("Edge weights must be non-negative"))
        
        # Ensure edge weights are symmetric
        for i in 1:n, j in 1:n
            if abs(edge_weights[i,j] - edge_weights[j,i]) > 1e-10
                throw(ArgumentError("Edge weights must be symmetric"))
            end
        end
        
        new(n, edge_weights, unary_costs, λ)
    end
end

ground_set_size(f::ImageSegmentationFunction) = f.n

function evaluate(f::ImageSegmentationFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    total_cost = 0.0
    
    # Unary terms (classification cost for selected pixels)
    for i in 1:f.n
        if S[i]
            total_cost += f.λ * f.unary_costs[i]
        end
    end
    
    # Pairwise terms (edge cuts between selected and unselected)
    for i in 1:f.n, j in i+1:f.n
        if S[i] != S[j]  # Edge is cut
            total_cost += f.edge_weights[i,j]
        end
    end
    
    return total_cost
end

"""
    FeatureSelectionFunction

Feature selection function for machine learning based on mutual information and redundancy.
Widely used in feature selection, active learning, and dimensionality reduction.

This function balances the informativeness of individual features with their redundancy,
following the principles of mutual information-based feature selection.
"""
struct FeatureSelectionFunction <: SubmodularFunction
    n::Int  # number of features
    relevance::Vector{Float64}      # relevance of each feature to target
    redundancy::Matrix{Float64}     # pairwise redundancy between features
    α::Float64  # balance between relevance and redundancy
    
    function FeatureSelectionFunction(n::Int, relevance::Vector{Float64}, 
                                    redundancy::Matrix{Float64}, α::Float64=0.5)
        n > 0 || throw(ArgumentError("Number of features must be positive, got $n"))
        length(relevance) == n || throw(ArgumentError("Relevance vector length must be $n, got $(length(relevance))"))
        size(redundancy) == (n, n) || throw(ArgumentError("Redundancy matrix must be $n×$n, got $(size(redundancy))"))
        0 <= α <= 1 || throw(ArgumentError("Alpha must be in [0,1], got $α"))
        all(r >= 0 for r in relevance) || throw(ArgumentError("Relevance scores must be non-negative"))
        all(r >= 0 for r in redundancy) || throw(ArgumentError("Redundancy scores must be non-negative"))
        
        # Ensure redundancy matrix is symmetric with zero diagonal
        for i in 1:n
            redundancy[i,i] == 0 || throw(ArgumentError("Redundancy matrix diagonal must be zero"))
            for j in 1:n
                if abs(redundancy[i,j] - redundancy[j,i]) > 1e-10
                    throw(ArgumentError("Redundancy matrix must be symmetric"))
                end
            end
        end
        
        new(n, relevance, redundancy, α)
    end
end

ground_set_size(f::FeatureSelectionFunction) = f.n

function evaluate(f::FeatureSelectionFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    if sum(S) == 0
        return 0.0
    end
    
    # Relevance term: sum of individual feature relevance
    relevance_term = sum(f.relevance[i] for i in 1:f.n if S[i])
    
    # Redundancy term: pairwise redundancy between selected features
    redundancy_term = 0.0
    selected_indices = findall(S)
    for i in selected_indices, j in selected_indices
        if i < j  # Avoid double counting
            redundancy_term += f.redundancy[i,j]
        end
    end
    
    # Return negative value for minimization (we want to maximize relevance - redundancy)
    return -(f.α * relevance_term - (1 - f.α) * redundancy_term)
end

"""
    DiversityFunction

Diversity maximization function based on pairwise distances/similarities.
Used in recommendation systems, active learning, and diverse subset selection.

This function promotes selecting diverse elements that are maximally different
from each other according to some distance/similarity metric.
"""
struct DiversityFunction <: SubmodularFunction
    n::Int  # number of elements
    similarities::Matrix{Float64}  # pairwise similarities
    
    function DiversityFunction(n::Int, similarities::Matrix{Float64})
        n > 0 || throw(ArgumentError("Number of elements must be positive, got $n"))
        size(similarities) == (n, n) || throw(ArgumentError("Similarities matrix must be $n×$n, got $(size(similarities))"))
        all(s >= 0 for s in similarities) || throw(ArgumentError("Similarities must be non-negative"))
        
        # Ensure similarities matrix is symmetric with zero diagonal
        for i in 1:n
            similarities[i,i] == 0 || throw(ArgumentError("Similarities matrix diagonal must be zero"))
            for j in 1:n
                if abs(similarities[i,j] - similarities[j,i]) > 1e-10
                    throw(ArgumentError("Similarities matrix must be symmetric"))
                end
            end
        end
        
        new(n, similarities)
    end
end

ground_set_size(f::DiversityFunction) = f.n

function evaluate(f::DiversityFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    selected_count = sum(S)
    if selected_count == 0
        return 0.0
    end
    
    # Diversity is measured as negative sum of pairwise similarities
    # (lower similarity = higher diversity)
    total_similarity = 0.0
    selected_indices = findall(S)
    
    for i in selected_indices, j in selected_indices
        if i < j  # Avoid double counting
            total_similarity += f.similarities[i,j]
        end
    end
    
    # Return negative for minimization (we want to minimize similarity = maximize diversity)
    return -total_similarity
end

"""
    SensorPlacementFunction

Sensor placement function for monitoring/coverage applications.
Used in environmental monitoring, surveillance, and IoT sensor networks.

This function models the coverage provided by sensors with overlapping sensing regions,
incorporating diminishing returns from redundant coverage.
"""
struct SensorPlacementFunction <: SubmodularFunction
    n::Int  # number of potential sensor locations
    coverage::BitMatrix  # coverage[location, target] = sensor covers target
    target_weights::Vector{Float64}  # importance weights for targets
    
    function SensorPlacementFunction(n::Int, coverage::BitMatrix, target_weights::Vector{Float64})
        n > 0 || throw(ArgumentError("Number of sensor locations must be positive, got $n"))
        size(coverage, 1) == n || throw(ArgumentError("Coverage matrix must have $n rows, got $(size(coverage, 1))"))
        length(target_weights) == size(coverage, 2) || throw(ArgumentError("Target weights length must match coverage columns"))
        all(w >= 0 for w in target_weights) || throw(ArgumentError("Target weights must be non-negative"))
        
        new(n, coverage, target_weights)
    end
end

ground_set_size(f::SensorPlacementFunction) = f.n

function evaluate(f::SensorPlacementFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    if sum(S) == 0
        return 0.0
    end
    
    n_targets = size(f.coverage, 2)
    total_coverage = 0.0
    
    # For each target, check if it's covered by any selected sensor
    for target in 1:n_targets
        covered = false
        for sensor in 1:f.n
            if S[sensor] && f.coverage[sensor, target]
                covered = true
                break
            end
        end
        if covered
            total_coverage += f.target_weights[target]
        end
    end
    
    # Return negative for minimization (we want to maximize coverage)
    return -total_coverage
end

"""
    InformationGainFunction

Information gain function for active learning and experimental design.
Models the expected information gain from selecting query points or experiments.

This function captures the diminishing returns property of information:
additional similar queries provide less new information.
"""
struct InformationGainFunction <: SubmodularFunction
    n::Int  # number of potential queries/experiments
    uncertainties::Vector{Float64}      # uncertainty/informativeness of each query
    correlations::Matrix{Float64}       # correlation between queries
    decay_factor::Float64              # how quickly information saturates
    
    function InformationGainFunction(n::Int, uncertainties::Vector{Float64}, 
                                   correlations::Matrix{Float64}, decay_factor::Float64=0.5)
        n > 0 || throw(ArgumentError("Number of queries must be positive, got $n"))
        length(uncertainties) == n || throw(ArgumentError("Uncertainties vector length must be $n"))
        size(correlations) == (n, n) || throw(ArgumentError("Correlations matrix must be $n×$n"))
        0 < decay_factor <= 1 || throw(ArgumentError("Decay factor must be in (0,1], got $decay_factor"))
        all(u >= 0 for u in uncertainties) || throw(ArgumentError("Uncertainties must be non-negative"))
        all(-1 <= c <= 1 for c in correlations) || throw(ArgumentError("Correlations must be in [-1,1]"))
        
        # Ensure correlation matrix is symmetric with unit diagonal
        for i in 1:n
            abs(correlations[i,i] - 1.0) < 1e-10 || throw(ArgumentError("Correlation matrix diagonal must be 1"))
            for j in 1:n
                if abs(correlations[i,j] - correlations[j,i]) > 1e-10
                    throw(ArgumentError("Correlation matrix must be symmetric"))
                end
            end
        end
        
        new(n, uncertainties, correlations, decay_factor)
    end
end

ground_set_size(f::InformationGainFunction) = f.n

function evaluate(f::InformationGainFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    selected_indices = findall(S)
    if isempty(selected_indices)
        return 0.0
    end
    
    # Compute total information gain with diminishing returns
    total_gain = 0.0
    
    for i in selected_indices
        # Base information from this query
        base_info = f.uncertainties[i]
        
        # Reduction due to correlation with already selected queries
        correlation_penalty = 0.0
        for j in selected_indices
            if i != j
                correlation_penalty += abs(f.correlations[i,j]) * f.uncertainties[j]
            end
        end
        
        # Apply decay factor to model diminishing returns
        effective_gain = base_info * (1 - f.decay_factor * correlation_penalty / base_info)
        effective_gain = max(0.0, effective_gain)  # Ensure non-negative
        
        total_gain += effective_gain
    end
    
    # Return negative for minimization (we want to maximize information gain)
    return -total_gain
end

# ============================================================================
# Economics and Auction Theory Submodular Functions
# ============================================================================

"""
    GrossSubstitutesFunction

Gross substitutes utility function from auction theory and economics.
This function satisfies the gross substitutes property: when prices of some goods increase,
demand for other goods does not decrease.

The gross substitutes property is equivalent to submodularity of the utility function,
making this fundamental in auction theory, matching markets, and mechanism design.
"""
struct GrossSubstitutesFunction <: SubmodularFunction
    n::Int  # number of goods/items
    valuations::Vector{Float64}         # individual valuations for each item
    complementarity::Matrix{Float64}    # positive complementarity effects
    substitutability::Matrix{Float64}   # negative substitutability effects
    
    function GrossSubstitutesFunction(n::Int, valuations::Vector{Float64}, 
                                    complementarity::Matrix{Float64}, 
                                    substitutability::Matrix{Float64})
        n > 0 || throw(ArgumentError("Number of goods must be positive, got $n"))
        length(valuations) == n || throw(ArgumentError("Valuations vector length must be $n"))
        size(complementarity) == (n, n) || throw(ArgumentError("Complementarity matrix must be $n×$n"))
        size(substitutability) == (n, n) || throw(ArgumentError("Substitutability matrix must be $n×$n"))
        all(v >= 0 for v in valuations) || throw(ArgumentError("Valuations must be non-negative"))
        all(c >= 0 for c in complementarity) || throw(ArgumentError("Complementarity effects must be non-negative"))
        all(s >= 0 for s in substitutability) || throw(ArgumentError("Substitutability effects must be non-negative"))
        
        # Ensure matrices are symmetric with zero diagonal
        for i in 1:n
            complementarity[i,i] == 0 || throw(ArgumentError("Complementarity matrix diagonal must be zero"))
            substitutability[i,i] == 0 || throw(ArgumentError("Substitutability matrix diagonal must be zero"))
            for j in 1:n
                abs(complementarity[i,j] - complementarity[j,i]) < 1e-10 || 
                    throw(ArgumentError("Complementarity matrix must be symmetric"))
                abs(substitutability[i,j] - substitutability[j,i]) < 1e-10 || 
                    throw(ArgumentError("Substitutability matrix must be symmetric"))
            end
        end
        
        new(n, valuations, complementarity, substitutability)
    end
end

ground_set_size(f::GrossSubstitutesFunction) = f.n

function evaluate(f::GrossSubstitutesFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    if sum(S) == 0
        return 0.0
    end
    
    # Individual valuations
    total_value = sum(f.valuations[i] for i in 1:f.n if S[i])
    
    # Complementarity effects (positive interactions)
    selected_indices = findall(S)
    for i in selected_indices, j in selected_indices
        if i < j  # Avoid double counting
            total_value += f.complementarity[i,j]
        end
    end
    
    # Substitutability effects (negative interactions, ensuring submodularity)
    for i in selected_indices, j in selected_indices
        if i < j  # Avoid double counting
            total_value -= f.substitutability[i,j]
        end
    end
    
    # Return negative for minimization (auction problems often maximize utility)
    return -total_value
end

"""
    AuctionRevenueFunction

Auction revenue function modeling the revenue from selling items to bidders.
Incorporates diminishing marginal returns and substitutability between items.

This function is submodular when items are gross substitutes, which is a fundamental
property in auction theory ensuring existence of competitive equilibria.
"""
struct AuctionRevenueFunction <: SubmodularFunction
    n::Int  # number of items/lots
    base_values::Vector{Float64}        # base value of each item
    bidder_preferences::Matrix{Float64} # bidder_preferences[item, bidder]
    competition_matrix::Matrix{Float64} # how items compete for bidders
    
    function AuctionRevenueFunction(n::Int, base_values::Vector{Float64}, 
                                  bidder_preferences::Matrix{Float64}, 
                                  competition_matrix::Matrix{Float64})
        n > 0 || throw(ArgumentError("Number of items must be positive, got $n"))
        length(base_values) == n || throw(ArgumentError("Base values vector length must be $n"))
        size(bidder_preferences, 1) == n || throw(ArgumentError("Bidder preferences must have $n rows"))
        size(competition_matrix) == (n, n) || throw(ArgumentError("Competition matrix must be $n×$n"))
        all(v >= 0 for v in base_values) || throw(ArgumentError("Base values must be non-negative"))
        all(p >= 0 for p in bidder_preferences) || throw(ArgumentError("Bidder preferences must be non-negative"))
        all(c >= 0 for c in competition_matrix) || throw(ArgumentError("Competition effects must be non-negative"))
        
        # Ensure competition matrix is symmetric with zero diagonal
        for i in 1:n
            competition_matrix[i,i] == 0 || throw(ArgumentError("Competition matrix diagonal must be zero"))
            for j in 1:n
                abs(competition_matrix[i,j] - competition_matrix[j,i]) < 1e-10 || 
                    throw(ArgumentError("Competition matrix must be symmetric"))
            end
        end
        
        new(n, base_values, bidder_preferences, competition_matrix)
    end
end

ground_set_size(f::AuctionRevenueFunction) = f.n

function evaluate(f::AuctionRevenueFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    if sum(S) == 0
        return 0.0
    end
    
    n_bidders = size(f.bidder_preferences, 2)
    total_revenue = 0.0
    
    # Base revenue from selected items
    total_revenue += sum(f.base_values[i] for i in 1:f.n if S[i])
    
    # Revenue from bidder competition
    for bidder in 1:n_bidders
        # Find highest preference among selected items for this bidder
        max_preference = 0.0
        for item in 1:f.n
            if S[item]
                max_preference = max(max_preference, f.bidder_preferences[item, bidder])
            end
        end
        total_revenue += max_preference
    end
    
    # Competition penalty (items competing for same bidders)
    selected_indices = findall(S)
    for i in selected_indices, j in selected_indices
        if i < j  # Avoid double counting
            total_revenue -= f.competition_matrix[i,j]
        end
    end
    
    # Return negative for minimization (we want to maximize revenue)
    return -total_revenue
end

"""
    MarketShareFunction

Market share function modeling how different products/strategies capture market segments.
Used in competitive analysis and strategic positioning.

This function exhibits submodularity through diminishing returns: additional similar
products provide decreasing marginal market share due to cannibalization.
"""
struct MarketShareFunction <: SubmodularFunction
    n::Int  # number of products/strategies
    segment_appeals::Matrix{Float64}    # appeal of each product to each segment
    segment_sizes::Vector{Float64}      # size/value of each market segment
    cannibalization::Matrix{Float64}    # how products cannibalize each other
    
    function MarketShareFunction(n::Int, segment_appeals::Matrix{Float64}, 
                               segment_sizes::Vector{Float64}, 
                               cannibalization::Matrix{Float64})
        n > 0 || throw(ArgumentError("Number of products must be positive, got $n"))
        size(segment_appeals, 1) == n || throw(ArgumentError("Segment appeals must have $n rows"))
        length(segment_sizes) == size(segment_appeals, 2) || 
            throw(ArgumentError("Segment sizes length must match appeals columns"))
        size(cannibalization) == (n, n) || throw(ArgumentError("Cannibalization matrix must be $n×$n"))
        all(a >= 0 for a in segment_appeals) || throw(ArgumentError("Segment appeals must be non-negative"))
        all(s >= 0 for s in segment_sizes) || throw(ArgumentError("Segment sizes must be non-negative"))
        all(c >= 0 for c in cannibalization) || throw(ArgumentError("Cannibalization effects must be non-negative"))
        
        # Ensure cannibalization matrix is symmetric with zero diagonal
        for i in 1:n
            cannibalization[i,i] == 0 || throw(ArgumentError("Cannibalization matrix diagonal must be zero"))
            for j in 1:n
                abs(cannibalization[i,j] - cannibalization[j,i]) < 1e-10 || 
                    throw(ArgumentError("Cannibalization matrix must be symmetric"))
            end
        end
        
        new(n, segment_appeals, segment_sizes, cannibalization)
    end
end

ground_set_size(f::MarketShareFunction) = f.n

function evaluate(f::MarketShareFunction, S::BitVector)
    length(S) == f.n || throw(ArgumentError("BitVector length must match ground set size"))
    
    if sum(S) == 0
        return 0.0
    end
    
    n_segments = length(f.segment_sizes)
    total_share = 0.0
    
    # Market share from each segment
    for segment in 1:n_segments
        # Find maximum appeal among selected products for this segment
        max_appeal = 0.0
        for product in 1:f.n
            if S[product]
                max_appeal = max(max_appeal, f.segment_appeals[product, segment])
            end
        end
        total_share += max_appeal * f.segment_sizes[segment]
    end
    
    # Cannibalization penalty
    selected_indices = findall(S)
    for i in selected_indices, j in selected_indices
        if i < j  # Avoid double counting
            total_share -= f.cannibalization[i,j]
        end
    end
    
    # Return negative for minimization (we want to maximize market share)
    return -total_share
end

# ============================================================================
# Factory Functions for AI/Vision and Economics Functions
# ============================================================================

"""
    create_image_segmentation(n::Int; edge_density::Float64=0.3, λ::Float64=1.0)

Create a random image segmentation function for computer vision applications.
"""
function create_image_segmentation(n::Int; edge_density::Float64=0.3, λ::Float64=1.0)
    # Random edge weights (similarity between adjacent pixels/superpixels)
    edge_weights = zeros(n, n)
    for i in 1:n, j in i+1:n
        if rand() < edge_density
            weight = rand() * 10.0  # Random similarity
            edge_weights[i,j] = weight
            edge_weights[j,i] = weight
        end
    end
    
    # Random unary costs (classification difficulty for each pixel)
    unary_costs = rand(n) * 5.0
    
    return ImageSegmentationFunction(n, edge_weights, unary_costs, λ)
end

"""
    create_feature_selection(n::Int; correlation_strength::Float64=0.5, α::Float64=0.6)

Create a random feature selection function for machine learning applications.
"""
function create_feature_selection(n::Int; correlation_strength::Float64=0.5, α::Float64=0.6)
    # Random feature relevance scores
    relevance = rand(n) * 10.0
    
    # Random redundancy matrix (correlation between features)
    redundancy = zeros(n, n)
    for i in 1:n, j in i+1:n
        corr = rand() * correlation_strength
        redundancy[i,j] = corr
        redundancy[j,i] = corr
    end
    
    return FeatureSelectionFunction(n, relevance, redundancy, α)
end

"""
    create_diversity_function(n::Int; similarity_strength::Float64=0.4)

Create a random diversity function for recommendation and selection applications.
"""
function create_diversity_function(n::Int; similarity_strength::Float64=0.4)
    # Random similarity matrix
    similarities = zeros(n, n)
    for i in 1:n, j in i+1:n
        sim = rand() * similarity_strength
        similarities[i,j] = sim
        similarities[j,i] = sim
    end
    
    return DiversityFunction(n, similarities)
end

"""
    create_sensor_placement(n::Int, m::Int; coverage_prob::Float64=0.3)

Create a random sensor placement function with n sensors and m targets.
"""
function create_sensor_placement(n::Int, m::Int; coverage_prob::Float64=0.3)
    # Random coverage matrix
    coverage = rand(n, m) .< coverage_prob
    
    # Random target importance weights
    target_weights = rand(m) * 5.0
    
    return SensorPlacementFunction(n, coverage, target_weights)
end

"""
    create_information_gain(n::Int; correlation_strength::Float64=0.4, decay::Float64=0.3)

Create a random information gain function for active learning applications.
"""
function create_information_gain(n::Int; correlation_strength::Float64=0.4, decay::Float64=0.3)
    # Random uncertainty/informativeness scores
    uncertainties = rand(n) * 10.0
    
    # Random correlation matrix
    correlations = zeros(n, n)
    for i in 1:n
        correlations[i,i] = 1.0
        for j in i+1:n
            corr = (rand() - 0.5) * 2 * correlation_strength  # Can be negative
            correlations[i,j] = corr
            correlations[j,i] = corr
        end
    end
    
    return InformationGainFunction(n, uncertainties, correlations, decay)
end

"""
    create_gross_substitutes(n::Int; substitutability_strength::Float64=0.3)

Create a random gross substitutes function from auction theory.
"""
function create_gross_substitutes(n::Int; substitutability_strength::Float64=0.3)
    # Random individual valuations
    valuations = rand(n) * 20.0
    
    # Small complementarity effects
    complementarity = zeros(n, n)
    for i in 1:n, j in i+1:n
        comp = rand() * 2.0  # Small positive complementarity
        complementarity[i,j] = comp
        complementarity[j,i] = comp
    end
    
    # Stronger substitutability effects to ensure gross substitutes property
    substitutability = zeros(n, n)
    for i in 1:n, j in i+1:n
        subst = rand() * substitutability_strength * 10.0
        substitutability[i,j] = subst
        substitutability[j,i] = subst
    end
    
    return GrossSubstitutesFunction(n, valuations, complementarity, substitutability)
end

"""
    create_auction_revenue(n::Int, m::Int; competition_strength::Float64=0.2)

Create a random auction revenue function with n items and m bidders.
"""
function create_auction_revenue(n::Int, m::Int; competition_strength::Float64=0.2)
    # Random base values for items
    base_values = rand(n) * 15.0
    
    # Random bidder preferences
    bidder_preferences = rand(n, m) * 10.0
    
    # Competition matrix (how items compete)
    competition_matrix = zeros(n, n)
    for i in 1:n, j in i+1:n
        comp = rand() * competition_strength * 5.0
        competition_matrix[i,j] = comp
        competition_matrix[j,i] = comp
    end
    
    return AuctionRevenueFunction(n, base_values, bidder_preferences, competition_matrix)
end

"""
    create_market_share(n::Int, m::Int; cannibalization_strength::Float64=0.25)

Create a random market share function with n products and m market segments.
"""
function create_market_share(n::Int, m::Int; cannibalization_strength::Float64=0.25)
    # Random segment appeals
    segment_appeals = rand(n, m) * 8.0
    
    # Random segment sizes
    segment_sizes = rand(m) * 10.0
    
    # Cannibalization matrix
    cannibalization = zeros(n, n)
    for i in 1:n, j in i+1:n
        cann = rand() * cannibalization_strength * 6.0
        cannibalization[i,j] = cann
        cannibalization[j,i] = cann
    end
    
    return MarketShareFunction(n, segment_appeals, segment_sizes, cannibalization)
end