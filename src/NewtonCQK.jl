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

##############################################################################
##############################################################################
##############################################################################

using LoopVectorization

export newton, tnewton

function initial_multiplier(x)
    return (1 - sum(x)) / length(x)
end

function tinitial_multiplier(x)
    totalx = treduce(+, x)
    return (1 - totalx) / length(x)
end

function newton_phi_step(x, λ, inds)
    T = eltype(x)
    phi = zero(T)
    deriv = zero(T)
    @turbo for i in inds
        new_x = x[i] + λ
        big = new_x >= 0
        deriv += big ? 1 : 0
        phi += big ? new_x : 0
    end
    return SA[phi, deriv]
end

function newton_phi(x::Vector{T}, λ::T) where {T<:Real}
    prec = 100 * eps(T)
    value, deriv = newton_phi_step(x, λ, 1:length(x))
    if abs(value - 1) < prec
        return SA[zero(T), deriv]
    else
        return SA[value - 1, deriv]
    end
end

function tnewton_phi(x::Vector{T}, λ::T) where {T<:Real}
    prec = 100 * eps(T)
    chunks = index_chunks(x; n=Threads.nthreads())
    tasks = map(chunks) do inds
        OhMyThreads.@spawn newton_phi_step(x, λ, inds)
    end
    value, deriv = sum(fetch, tasks)
    if abs(value - 1) < prec
        return SA[zero(T), deriv]
    else
        return SA[value - 1, deriv]
    end
end

function dual2primal!(sol, x, λ, inds=1:length(x))
    @turbo for i in inds
        pre_sol = x[i] + λ
        sol[i] = pre_sol ≥ 0 ? pre_sol : 0
    end
end

function dual2primal(x::Vector{T}, λ::T) where {T<:Real}
    sol = similar(x)
    dual2primal!(sol, x, λ)
    return sol
end

function tdual2primal(x::Vector{T}, λ::T) where {T<:Real}
    sol = similar(x)
    chunks = index_chunks(x; n=Threads.nthreads())
    @sync map(chunks) do inds
        OhMyThreads.@spawn dual2primal!(sol, x, λ, inds)
    end
    return sol
end

"""
Naive semismooth Newton method to project x onto the unit Simplex.

There is no globalization. It assumes it will converge. 
"""
function pre_newton(x, maxiters, initial_multiplier, newton_phi, dual2primal)
    # Initialization
    T = eltype(x)
    λ = initial_multiplier(x)
    φ, φ′ = newton_phi(x, λ)
    iter = 0

    # Newton loop
    while φ != 0 && iter <= maxiters
        old_λ = λ
        λ -= φ / φ′
        if λ == old_λ
            φ = zero(T)
            break
        end
        φ, φ′ = newton_phi(x, λ)
        iter += 1
    end

    # Return resuts
    if φ == 0
        sol = dual2primal(x, λ)
        return sol, iter
    else
        return zero(x), min(-iter, -1)
    end
end

function newton(x; maxiters=100)
    return pre_newton(x, maxiters, initial_multiplier, newton_phi, dual2primal)
end
function tnewton(x; maxiters=100)
    return pre_newton(x, maxiters, tinitial_multiplier, tnewton_phi, tdual2primal)
end

function random_x(n)
    fillin = max(1, Int(floor(rand()^2 * n)))
    inds = rand(1:n, fillin)
    x = zeros(n)
    x[inds] = rand(fillin)
    return x
end

end
