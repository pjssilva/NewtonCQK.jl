###############################
# PROJECTION y ONTO THE l1 BALL
# x; sum_i |x[i]| <= r
###############################

# Compute solution as a dense vector
function l1ball_dense_solution!(
    y::Vector{T}, λ::T, sol::Vector{T}, chunks::Vector{C}, nchunks
) where {T<:AbstractFloat,C<:AbstractChunk}
    if nchunks == 1
        fill!(sol, zero(T))
    else
        OhMyThreads.@tasks for i in eachindex(sol)
            @set nchunks = nchunks
            @inbounds sol[i] = zero(T)
        end
    end
    for chunk in chunks
        @inbounds for i in (chunk.start):(chunk.final)
            ii = chunk.active[i]
            new_x = abs(y[ii]) + λ
            if new_x > zero(T)
                sol[ii] = copysign(new_x, y[ii])
            end
        end
    end
    return nothing
end

# Compute solution as a new sparse vector
function l1ball_sparse_solution(
    y::Vector{T}, λ::T, chunks::Vector{C}
) where {T<:AbstractFloat,C<:AbstractChunk}
    inds = UInt[]
    vals = T[]

    for chunk in chunks
        @inbounds for i in (chunk.start):(chunk.final)
            ii = chunk.active[i]
            new_x = abs(y[ii]) + λ
            if new_x > zero(T)
                push!(vals, copysign(new_x, y[ii]))
                push!(inds, ii)
            end
        end
    end
    return sparsevec(inds, vals, length(y))
end

"""
    iter = l1ball_proj!(sol, y; r=1.0, maxiters=100, nchunks=Threads.nthreads(), x0=[], chunks=Chunk[])

Parallel semismooth Newton method to project `y` onto the l1-ball 
`x: || x ||_1 <= r`.

The projection in written in `sol`, which must have the same size and type of
`y`. `iter` returns the number of Newton steps used. It indicates failure if
negative.

It is possible to pre-allocate, for efficiency, the workspace using:

`chunks = initialize_chunks(n; nchunks=Threads.nthreads())`

Then, it can be used in subsequent executions of `simplex_proj`:

`iter = simplex_proj(sol, y; chunks=chunks)`

In this case, `nchunks` in `simplex_proj` will be ignored and the value
used when creating the workspace will be used instead.

Obs: `sol` has to be a different vector that `y`
"""
function l1ball_proj!(
    sol::Vector{T},
    y::Vector{T};
    r=one(T),
    maxiters=100,
    nchunks=Threads.nthreads(),
    chunks::Vector{C}=AbstractChunk[],
    x0::Vector{T}=T[]
) where {T<:AbstractFloat,C<:AbstractChunk}
    # Asserts that the user is not doing aliasing between input and output
    @assert sol !== y "sol and y vectors cannot be the same"

    if isempty(chunks)
        chunks = initialize_chunks(DynamicChunk, length(y); nchunks=nchunks)
    else
        nchunks = length(chunks)
    end

    # abs(x0) and abs(y) as a lazy vector
    absy = BroadcastVector(abs, y)
    absx0 = BroadcastVector(abs, x0)

    # Project onto the simplex
    λ, iter, flag = simplex_newton(absy, absx0, r, chunks, maxiters)

    l1ball_dense_solution!(y, λ, sol, chunks, nchunks)
    return iter, flag
end

"""
    sol, iter = l1ball_proj(y; r=1.0, maxiters=100, nchunks=Threads.nthreads(), x0=[], chunks=Chunk[])

Variation of `simplex_proj!`, that allocates a new vector for the solution,
returning it.
"""
function l1ball_proj(
    y::Vector{T};
    r=one(T),
    maxiters=100,
    nchunks=Threads.nthreads(),
    chunks::Vector{C}=AbstractChunk[],
    x0::Vector{T}=T[]
) where {T<:AbstractFloat,C<:AbstractChunk}
    sol = similar(y)
    iter, flag = l1ball_proj!(
        sol, y; r=r, maxiters=maxiters, nchunks=nchunks, chunks=chunks, x0=x0
    )
    return sol, iter, flag
end

"""
    sol, iter = spl1ball_proj(y; r=1.0, maxiters=100, nchunks=Threads.nthreads(), x0=[], chunks=Chunk[])

Parallel semismooth Newton method to project `y` onto the l1-ball 
`x: || x ||_1 <= r`.

The projection is returned as the sparse vector `sol`, that has the same
`eltype` as y. `iter` returns the number of Newton steps used. It indicates
failure if negative.
"""
function spl1ball_proj(
    y::Vector{T};
    r=one(T),
    maxiters=100,
    nchunks=Threads.nthreads(),
    chunks::Vector{C}=AbstractChunk[],
    x0::Vector{T}=T[]
) where {T<:AbstractFloat,C<:AbstractChunk}
    if isempty(chunks)
        chunks = initialize_chunks(DynamicChunk, length(y); nchunks=nchunks)
    end

    # abs(x0) and abs(y) as a lazy vector
    absy = BroadcastVector(abs, y)
    absx0 = BroadcastVector(abs, x0)

    # Project onto the simplex
    λ, iter, flag = simplex_newton(absy, absx0, r, chunks, maxiters)

    return l1ball_sparse_solution(y, λ, chunks), iter, flag
end
