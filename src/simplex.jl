###############################
# PROJECTION y ONTO THE SIMPLEX
# x; sum_i x[i] == r,  x >= 0
###############################

# Functions to compute the norm1 of y if needed. We are using the fact the
# original projections onto the simplex are based on (concrete) Vectors while
# projection onto the l1-ball use BroadcastVector to differentiate between the
# two.
@inline function fetch_accum!(total::Base.RefValue{T}, y::Vector{T}, i) where {T}
    return y[i]
end

@inline function fetch_accum!(total::Base.RefValue{T}, y::BroadcastVector{T}, i) where {T}
    total[] += y[i]
    return y[i]
end

@inline function isfeasible(total::T, r, v::Vector{T}) where {T}
    return false
end

@inline function isfeasible(total::T, r, v::BroadcastVector{T}) where {T}
    return total <= r
end

@inline function ispositive(yvali::T, v::Vector{T}) where {T}
    return true
end

@inline function ispositive(yvali::T, v::BroadcastVector{T}) where {T}
    return yvali > zero(T)
end

# To initialize λ when x0 is not provided
function simplex_init(
    y::AbstractVector{T}, r, chunk::AbstractChunk
) where {T<:AbstractFloat}
    wait_idx = 1
    len = 1
    k = addfirst!(chunk)
    @inbounds sumy = y[chunk.ystart]
    λ = r - sumy
    total = Ref(sumy)
    @inbounds for i in (chunk.ystart + 1):(chunk.yfinal)
        yvali = fetch_accum!(total, y, i)
        new_x = yvali + λ
        if (new_x > zero(T)) && ispositive(yvali, y)
            len += 1
            k = addnext!(chunk, k, i)
            sumy += yvali
            λ -= new_x / len
            if yvali + λ >= r
                # End of the waiting list
                wait_idx = lenactive(chunk, k)
                len = 1
                sumy = yvali
                λ = r - sumy
            end
        end
    end
    chunk.final = lenactive(chunk, k)

    # At this point, chunk.active[1:wait_idx-1] is the waiting list
    # Processes it in reverse order, reusing the indices
    # At the end, k will represent the position of the leftmost index in chunk.active
    k = wait_idx
    @inbounds for i in (wait_idx - 1):-1:1
        ii = chunk.active[i]
        new_x = y[ii] + λ
        if new_x > zero(T)
            k -= 1
            chunk.active[k] = ii
            len += 1
            sumy += y[ii]
            λ -= new_x / len
        end
    end
    chunk.start = k

    return SA[total[], sumy, len]
end

# To initialize λ when x0 is provided. It is assumed that x0 >= 0.
function simplex_init(
    y::AbstractVector{T}, x0::AbstractVector{T}, r, chunk::AbstractChunk
) where {T<:AbstractFloat}
    wait_idx = 1
    len = 0
    k = 0
    sumy = zero(T)
    λ = r
    total = Ref(sumy)
    @inbounds for i in (chunk.ystart):(chunk.yfinal)
        yvali = fetch_accum!(total, y, i)
        new_x = yvali + λ
        if (new_x > zero(T)) && ispositive(yvali, y)
            k = addnext!(chunk, k, i)
            if x0[i] > zero(T)        # λ is computed over the indices i s.t. x0[i] > 0
                len += 1
                sumy += yvali
                λ -= new_x / len
                new_x = yvali + λ
            end
            if new_x >= r
                # End of the waiting list
                wait_idx = lenactive(chunk, k)
                if x0[i] > zero(T)
                    len = 1
                    sumy = yvali
                    λ = r - sumy
                else
                    len = 0
                    sumy = zero(T)
                    λ = r
                end
            end
        end
    end
    chunk.final = lenactive(chunk, k)

    # At this point, chunk.active[1:wait_idx-1] is the waiting list
    # Processes it in reverse order, reusing the indices
    # At the end, k will represent the position of the leftmost index in chunk.active
    k = wait_idx
    @inbounds for i in (wait_idx - 1):-1:1
        ii = chunk.active[i]
        new_x = y[ii] + λ
        if new_x > zero(T)
            k -= 1
            chunk.active[k] = ii
            if x0[i] > zero(T)
                len += 1
                sumy += y[ii]
                λ -= new_x / len
            end
        end
    end
    chunk.start = k

    return SA[total[], sumy, len]
end

# Compute phi and phi' (possibly rhi - r < 0)
function simplex_phi_step(
    y::AbstractVector{T}, λ::T, r, chunk::AbstractChunk
) where {T<:AbstractFloat}
    phi = zero(T)
    fixidx = 0

    # Run the vector chunk.active[chunk.start:chunk:final]
    @inbounds for i in (chunk.start):(chunk.final)
        ii = chunk.active[i]
        new_x = y[ii] + λ
        if new_x >= zero(T)
            phi += new_x
            if phi - r > zero(T)
                # from this point, the condition for fixing variables holds
                fixidx = i
                break
            end
        end
    end

    if fixidx > 0
        # Continue discardng fixed variables
        k = fixidx
        @inbounds for i in (fixidx + 1):(chunk.final)
            ii = chunk.active[i]
            new_x = y[ii] + λ
            if new_x >= zero(T)
                phi += new_x
                k += 1
                chunk.active[k] = ii
            end
        end

        # Update the final position in chunk.active
        chunk.final = k

        # Search for non-fixed variables in the first run
        k = fixidx
        @inbounds for i in (fixidx - 1):-1:(chunk.start)
            ii = chunk.active[i]
            new_x = y[ii] + λ
            if new_x >= zero(T)
                k -= 1
                chunk.active[k] = ii
            end
        end

        # Update the start position in chunk.active
        chunk.start = k
    end

    return SA[phi, chunk.final - chunk.start - 1]
end

# Compute phi and phi' considering rhi - r >= 0
function simplex_phi_step(
    y::AbstractVector{T}, λ::T, chunk::AbstractChunk
) where {T<:AbstractFloat}
    phi = zero(T)

    # Run the vector chunk.active[chunk.start:chunk:final]
    # The condition for fixing variables (rhi - r >= 0) holds even before computing phi
    k = chunk.start
    @inbounds for i in (chunk.start):(chunk.final)
        ii = chunk.active[i]
        new_x = y[ii] + λ
        if new_x >= zero(T)
            phi += new_x
            chunk.active[k] = ii
            k += 1
        end
    end

    # Update the final position in chunk.active
    chunk.final = k - 1

    return SA[phi, k - chunk.start]
end

@inline function simplex_phi(
    y::AbstractVector{T}, λ::T, r, prefix::Bool, chunks::Vector{C}
) where {T<:AbstractFloat,C<:AbstractChunk}
    if prefix
        phi, deriv = altmapreduce(
            c -> simplex_phi_step(y, λ, c), .+, chunks; init=(zero(T), zero(T))
        )
    else
        phi, deriv = altmapreduce(
            c -> simplex_phi_step(y, λ, r, c), .+, chunks; init=(zero(T), zero(T))
        )
    end
    return phi, deriv
end

function simplex_newton(
    y::AbstractVector{T}, x0::AbstractVector{T}, r, chunks::Vector{C}, maxiters
) where {T<:AbstractFloat,C<:AbstractChunk}
    T0 = zero(T)
    if isempty(x0)
        # Initialize λ and fix variables
        total, sumy, len = altmapreduce(
            chunk -> simplex_init(y, r, chunk), .+, chunks; init=(T0, T0, T0)
        )

        λ = (r - sumy) / len

        if abs(λ) < eps(T) || isfeasible(total, r, y)
            # y is the solution!
            # iter = 0 indicates the solution need not be re-computed
            return T0, 0, true
        end

        φ, φ′ = simplex_phi(y, λ, r, true, chunks)
    else
        total, sumy, len = altmapreduce(
            chunk -> simplex_init(y, x0, r, chunk), .+, chunks; init=(T0, T0, T0)
        )

        if isfeasible(total, r, y)
            return T0, 0, true
        end

        if len > 0
            λ = (r - sumy) / len
        else
            # If x0 = 0 then return a very simple λ >= min{-y_i}.
            # This differs from the article because we do not store the set J in simplex_init.
            # As this is a rare situation, we prefer not to pay the price of additional computations.
            λ = max(r / length(y), -y[1])
        end

        φ, φ′ = simplex_phi(y, λ, r, false, chunks)
    end
    chunks = compress_chunks(chunks)

    # Newton loop
    iter = 1
    solved = false
    while (iter < maxiters)
        δ = (φ - r) / φ′
        old_λ = λ
        λ -= δ
        if (δ < eps(T)) || (old_λ == λ)
            # If δ is too small or it does not modify λ, stop. In both cases, φ-r ≈ 0.
            solved = true
            break
        end
        iter += 1
        # compute φ and φ' for the next iteration
        φ, φ′ = simplex_phi(y, λ, r, true, chunks)
        chunks = compress_chunks(chunks)
    end

    return λ, iter, solved
end

# Compute solution as a dense vector
function simplex_dense_solution!(
    y::Vector{T}, λ::T, sol::Vector{T}, chunks::Vector{C}, numthreads
) where {T<:AbstractFloat,C<:AbstractChunk}
    if numthreads == 1
        fill!(sol, zero(T))
    else
        OhMyThreads.@tasks for i in eachindex(sol)
            @set nchunks = numthreads
            @inbounds sol[i] = zero(T)
        end
    end
    for chunk in chunks
        @inbounds for i in (chunk.start):(chunk.final)
            ii = chunk.active[i]
            new_x = y[ii] + λ
            if new_x > zero(T)
                sol[ii] = new_x
            end
        end
    end
    return nothing
end

# Compute solution as a new sparse vector
function simplex_sparse_solution(
    y::Vector{T}, λ::T, chunks::Vector{C}
) where {T<:AbstractFloat,C<:AbstractChunk}
    inds = UInt[]
    vals = T[]

    for chunk in chunks
        @inbounds for i in (chunk.start):(chunk.final)
            ii = chunk.active[i]
            new_x = y[ii] + λ
            if new_x > zero(T)
                push!(vals, new_x)
                push!(inds, ii)
            end
        end
    end
    return sparsevec(inds, vals, length(y))
end

"""
    iter = simplex_proj!(sol, y; r=1.0, maxiters=100, numthreads=Threads.nthreads(), x0=[], chunks=Chunk[])

Parallel semismooth Newton method to project `y` onto the simplex `x: sum_i
x[i] == r, x >= 0`.

The projection in written in `sol`, which must have the same size and type of
`y`. `iter` returns the number of Newton steps used. It indicates failure if
negative.

It is possible to pre-allocate, for efficiency, the workspace using:

`chunks = initialize_chunks(n; numthreads=Threads.nthreads())`

Then, it can be used in subsequent executions of `simplex_proj!`:

`iter = simplex_proj!(sol, y; chunks=chunks)`

In this case, `numthreads` in `simplex_proj!` will be ignored and the value
used when creating the workspace will be used instead.

Obs: `sol` has to be a different vector that `y`
"""
function simplex_proj!(
    sol::Vector{T},
    y::Vector{T};
    r=one(T),
    maxiters=100,
    numthreads=Threads.nthreads(),
    chunks::Vector{C}=AbstractChunk[],
    x0::Vector{T}=T[]
)::Int where {T<:AbstractFloat,C<:AbstractChunk}
    # Asserts that the user is not doing aliasing between input and output
    @assert sol !== y "sol and y vectors cannot be the same"

    if isempty(chunks)
        chunks = initialize_chunks(DynamicChunk, length(y); numthreads=numthreads)
    else
        numthreads = length(chunks)
    end
    λ, iter, solved = simplex_newton(y, x0, r, chunks, maxiters)

    if solved
        simplex_dense_solution!(y, λ, sol, chunks, numthreads)
        return iter
    else
        return min(-iter, -1)
    end
end

"""
    sol, iter = simplex_proj(y; r=1.0, maxiters=100, numthreads=Threads.nthreads(), x0=[], chunks=Chunk[])

Variation of `simplex_proj!`, that allocates a new vector for the solution,
returning it.
"""
function simplex_proj(
    y::Vector{T};
    r=one(T),
    maxiters=100,
    numthreads=Threads.nthreads(),
    chunks::Vector{C}=AbstractChunk[],
    x0::Vector{T}=T[]
)::Tuple{Vector{T},Int} where {T<:AbstractFloat,C<:AbstractChunk}
    sol = similar(y)
    iter = simplex_proj!(
        sol, y; r=r, maxiters=maxiters, numthreads=numthreads, chunks=chunks, x0=x0
    )
    return sol, iter
end

"""
    sol, iter = spsimplex_proj(y; r=1.0, maxiters=100, numthreads=Threads.nthreads(), x0=[], chunks=Chunk[])

Parallel semismooth Newton method to project `y` onto the simplex `x: sum_i
x[i] == r, x >= 0`.

The projection is returned as the sparse vector `sol`, that has the same
`eltype` as y. `iter` returns the number of Newton steps used. It indicates
failure if negative.
"""
function spsimplex_proj(
    y::Vector{T};
    r=one(T),
    maxiters=100,
    numthreads=Threads.nthreads(),
    chunks::Vector{C}=AbstractChunk[],
    x0::Vector{T}=T[]
)::Tuple{SparseVector{T,UInt},Int} where {T<:AbstractFloat,C<:AbstractChunk}
    if isempty(chunks)
        chunks = initialize_chunks(DynamicChunk, length(y); numthreads=numthreads)
    end

    λ, iter, solved = simplex_newton(y, x0, r, chunks, maxiters)

    if solved
        # return SparseVector(length(y), UInt[], T[]), iter
        return simplex_sparse_solution(y, λ, chunks), iter
    else
        return SparseVector(length(y), UInt[], T[]), min(-iter, -1)
    end
end
