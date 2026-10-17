"""
Parallel implementations of the Semismooth Newton methods to solve Continuous
Quadratic Knapsack problems.
"""

module NewtonCQK

using StaticArrays
using Polyester
using LinearAlgebra
using Statistics
using SparseArrays
using Base.Threads
using CUDA
using LazyArrays

# Memory allocation
export initialize_chunks, AbstractChunk, FixedChunk, DynamicChunk
# Continuous quadratic knapsack
export cqk, cqk!, CQKProblem, create_cqkproblem
# Projection onto simplex
export simplex_proj, simplex_proj!, spsimplex_proj
# Projection onto l1 ball
export l1ball_proj, l1ball_proj!, spl1ball_proj

include("alloc.jl")

include("cqk.jl")
include("simplex.jl")
include("l1ball.jl")

include("cucqk.jl")
include("cusimplex.jl")
include("cul1ball.jl")

# Mapreduce
# Threaded with Polyester, a lightweight threading system: the chunks are few and each
# may hold little work, so spawn cost can dominate. `buf` is needed because `@batch`
# carries no reduction. The `::R` annotations keep the return type concrete -- the
# branch is resolved at runtime, so without them every caller boxes the result.
@inline function altmapreduce(f, op, it; init)
    R = Base.promote_op(f, eltype(it))
    nchunks = length(it)
    if nchunks == 1
        @inbounds return f(it[1])::R
    end
    buf = Vector{R}(undef, nchunks)
    Polyester.@batch for i in 1:nchunks
        @inbounds buf[i] = f(it[i])::R
    end
    acc = init
    @inbounds for i in 1:nchunks
        acc = op(acc, buf[i])
    end
    return acc::R
end

end
