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
using LoopVectorization
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
# Continuous quadratic knapsack (CUDA version)
export cucqk, cucqk!, cuCQKProblem, cucreate_cqkproblem
# Projection onto simplex (CUDA version)
export cusimplex_proj, cusimplex_proj!
# Projection onto l1 ball (CUDA version)
export cul1ball_proj, cul1ball_proj!

include("alloc.jl")

include("cqk.jl")
include("simplex.jl")
include("l1ball.jl")

include("cucqk.jl")
include("cusimplex.jl")
include("cul1ball.jl")

# Mapreduce
@inline function altmapreduce(f, op, it; init)
    if length(it) == 1
        @inbounds return f(it[1])
    else
        return OhMyThreads.tmapreduce(
            f, op, it; init=init, scheduler=:static, nchunks=length(it)
        )
    end
end

# Foreach
@inline function altforeach(f!, it)
    if length(it) == 1
        @inbounds return f!(it[1])
    else
        return OhMyThreads.tforeach(f!, it; scheduler=:static, nchunks=length(it))
    end
end
