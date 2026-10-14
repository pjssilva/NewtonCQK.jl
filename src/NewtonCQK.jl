"""
Parallel implementations of the Semismooth Newton methods to solve Continuous
Quadratic Knapsack problems.
"""

module NewtonCQK

using StaticArrays
using OhMyThreads
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
# The `::R` annotations are required: `tmapreduce` infers as `Any` and the branch is
# decided at runtime, so without them this returns `Any` and every caller boxes its
# result, even for a single chunk.
@inline function altmapreduce(f, op, it; init)
    R = Base.promote_op(f, eltype(it))
    if length(it) == 1
        @inbounds return f(it[1])::R
    else
        return OhMyThreads.tmapreduce(
            f, op, it; init=init, scheduler=:static, nchunks=length(it)
        )::R
    end
end

# Foreach
# Both branches must return `nothing`: `tforeach` does, so propagating `f!`'s value
# instead would make this infer as a `Union` and every caller box the result.
@inline function altforeach(f!, it)
    if length(it) == 1
        @inbounds f!(it[1])
    else
        OhMyThreads.tforeach(f!, it; scheduler=:static, nchunks=length(it))
    end
    return nothing
end

end
