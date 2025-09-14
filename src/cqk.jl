###########################################################
# SOLVE CONTINUOUS QUADRATIC KNAPSACK PROBLEMS OF THE FORM
#
# min_x 1/2*x'*D*x - a'*t  s.t.  b'*x = r,  l <= x <= u
#
# It is supposed that b > 0 and D = Diagonal(d) with d > 0.
###########################################################

"""
    CQKProblem{T<:AbstractFloat,V<:AbstractVector{T}}

A struct representing a continuous quadratic knapsack problem.

min_x 1/2*x'*D*x - a'*t  s.t.  b'*x = r,  l <= x <= u

# Fields
- `d::V`: The diagonal elements of the positive definite matrix `D`.
- `a::V`: The vector `a` in the objective function.
- `b::V`: The vector `b` in the constraint `b'x = r`.
- `r::T`: The right-hand side of the constraint `b'x = r`.
- `l::V`: The lower bounds for the variables `x`.
- `u::V`: The upper bounds for the variables `x`.

# Assumptions
- `b > 0` (component-wise).
- `D = Diagonal(d)` with `d > 0` (component-wise).
"""
struct CQKProblem{T<:AbstractFloat,V<:AbstractVector{T}}
    d::V
    a::V
    b::V
    r::T
    l::V
    u::V
end

# Initialize λ when x0 is not provided
function cqk_init(
    P::CQKProblem{T,V}, chunk::FixedChunk
) where {T<:AbstractFloat,V<:Vector{T}}
    s = zero(T)
    q = zero(T)
    k = 0

    @inbounds for i in (chunk.ystart):(chunk.yfinal)
        if (P.d[i] <= zero(T)) || (P.b[i] <= zero(T)) || (P.l[i] > P.u[i])
            # Invalid data
            return SA[zero(T), T(Inf)]
        end

        b_div_d = P.b[i] / P.d[i]

        s += P.a[i] * b_div_d
        q += P.b[i] * b_div_d
        k += 1
        chunk.active[k] = i
    end

    # Update start and final positions in chunk.active
    chunk.start = 1
    chunk.final = k

    return SA[s, q]
end

# Initialize λ when x0 is provided
function cqk_init(
    P::CQKProblem{T,V}, x0::Vector{T}, chunk::FixedChunk
) where {T<:AbstractFloat,V<:Vector{T}}
    r_diff = zero(T)
    s = zero(T)
    q = zero(T)
    k = 0

    @inbounds for i in (chunk.ystart):(chunk.yfinal)
        if (P.d[i] <= zero(T)) || (P.b[i] <= zero(T)) || (P.l[i] > P.u[i])
            # Invalid data
            return SA[zero(T), T(Inf), zero(T)]
        end

        if x0[i] <= P.l[i]
            r_diff -= P.b[i] * P.l[i]
        elseif x0[i] >= P.u[i]
            r_diff -= P.b[i] * P.u[i]
        else
            b_div_d = P.b[i] / P.d[i]
            s += P.a[i] * b_div_d
            q += P.b[i] * b_div_d
        end
        k += 1
        chunk.active[k] = i
    end

    # Update start and final positions in chunk.active
    chunk.start = 1
    chunk.final = k

    if q == zero(T)
        r_diff = zero(T)
        @inbounds for i in (chunk.start):(chunk.final)
            b_div_d = P.b[i] / P.d[i]
            s += P.a[i] * b_div_d
            q += P.b[i] * b_div_d
        end
    end

    return SA[s, q, r_diff]
end

# Compute phi and phi'
function cqk_phi_step(
    P::CQKProblem{T,V}, x::Vector{T}, λ::T, chunk::FixedChunk
) where {T<:AbstractFloat,V<:Vector{T}}
    phi = zero(T)
    deriv = zero(T)
    absphi = zero(T)

    @inbounds for i in (chunk.start):(chunk.final)
        ii = chunk.active[i]
        new_x = (P.b[ii] * λ + P.a[ii]) / P.d[ii]
        if new_x >= P.l[ii]
            if new_x <= P.u[ii]
                # As in the implementation of Comineti, Mascarenhas and Silva,
                # we do not differ from negative or positive derivatives
                deriv += P.b[ii] * (P.b[ii] / P.d[ii])
            else
                new_x = P.u[ii]
            end
        else
            new_x = P.l[ii]
        end
        x[ii] = new_x
        phi_ii = P.b[ii] * new_x
        phi += phi_ii
        absphi += abs(phi_ii)
    end

    return phi, deriv, absphi
end

@inline function cqk_phi(
    P::CQKProblem{T,V}, x::Vector{T}, λ::T, chunks::Vector{FixedChunk}
) where {T<:AbstractFloat,V<:Vector{T}}
    return let λ = λ
        altmapreduce(
            c -> cqk_phi_step(P, x, λ, c), .+, chunks; init=(zero(T), zero(T), zero(T))
        )
    end
end

function breakpoint_to_the_right(
    P::CQKProblem{T,V}, λ::T, chunk::FixedChunk
) where {T<:AbstractFloat,V<:Vector{T}}
    closest_bkp = T(Inf)
    @inbounds for i in (chunk.start):(chunk.final)
        ii = chunk.active[i]
        bkp = (P.d[ii] * P.l[ii] - P.a[ii]) / P.b[ii]
        if (bkp > λ) && (bkp < closest_bkp)
            closest_bkp = bkp
        end
    end
    return closest_bkp
end

function breakpoint_to_the_left(
    P::CQKProblem{T,V}, λ::T, chunk::FixedChunk
) where {T<:AbstractFloat,V<:Vector{T}}
    closest_bkp = T(-Inf)
    @inbounds for i in (chunk.start):(chunk.final)
        ii = chunk.active[i]
        bkp = (P.d[ii] * P.u[ii] - P.a[ii]) / P.b[ii]
        if (bkp < λ) && (bkp > closest_bkp)
            closest_bkp = bkp
        end
    end
    return closest_bkp
end

@inline function closest_breakpoint(
    P::CQKProblem{T,V}, λ::T, phi_minus_r::T, chunks::Vector{FixedChunk}
) where {T<:AbstractFloat,V<:Vector{T}}
    if phi_minus_r < zero(T)
        return let λ = λ
            altmapreduce(c -> breakpoint_to_the_right(P, λ, c), min, chunks; init=(zero(T)))
        end
    else
        return let λ = λ
            altmapreduce(c -> breakpoint_to_the_left(P, λ, c), max, chunks; init=(zero(T)))
        end
    end
end

@inline function secant_step(
    lo_λ::T, up_λ::T, lo_phi::T, up_phi::T
) where {T<:AbstractFloat}
    λ = (lo_λ - lo_phi * (up_λ - lo_λ)) / (up_phi - lo_phi)
    # if the secant step fails, try the midpoint
    if (λ >= up_λ) || (λ <= lo_λ)
        λ = (up_λ + lo_λ)/T(2)
    end
    return λ
end

function fix_variables_l(
    P::CQKProblem{T,V}, x::Vector{T}, chunk::FixedChunk
) where {T<:AbstractFloat,V<:Vector{T}}
    k = chunk.start - 1
    r_diff = 0.0 
    @inbounds for i in (chunk.start):(chunk.final)
        ii = chunk.active[i]
        if x[ii] <= P.l[ii]
            x[ii] = P.l[ii]
            r_diff -= P.b[ii] * x[ii]
        else
            k += 1
            chunk.active[k] = ii
        end
    end
    chunk.final = k
    return r_diff
end

function fix_variables_u(
    P::CQKProblem{T,V}, x::Vector{T}, chunk::FixedChunk
) where {T<:AbstractFloat,V<:Vector{T}}
    k = chunk.start - 1
    r_diff = 0.0
    @inbounds for i in (chunk.start):(chunk.final)
        ii = chunk.active[i]
        if x[ii] >= P.u[ii]
            x[ii] = P.u[ii]
            r_diff -= P.b[ii] * x[ii]
        else
            k += 1
            chunk.active[k] = ii
        end
    end
    chunk.final = k
    return r_diff
end

function cqk_newton(
    P::CQKProblem{T,V}, x0::Vector{T}, x::Vector{T}, chunks::Vector{FixedChunk}, maxiters
) where {T<:AbstractFloat,V<:Vector{T}}
    T0 = zero(T)
    lo_λ = T(-Inf)
    up_λ = T(Inf)
    lo_φ = lo_λ
    up_φ = up_λ
    r = Float64(P.r)
    fixed_low = 0.0
    fixed_up = 0.0

    # λ initialization
    if isempty(x0)
        s, q = altmapreduce(c -> cqk_init(P, c), .+, chunks; init=(T0, T0))
        λ = (T(r) - s) / q
    else
        s, q, r_diff = altmapreduce(c -> cqk_init(P, x0, c), .+, chunks; init=(T0, T0, T0))
        λ = (T(r) + r_diff - s) / q
    end

    # q is Inf if data is inconsistent or an infeasibility was identified
    if isinf(q)
        return T0, 0, 1
    end

    flag = 3

    # Newton loop
    iter = 0
    while (iter < maxiters)
        iter += 1

        # Compute φ-r and φ'
        φ, φ′, abs_φ = cqk_phi(P, x, λ, chunks)
        φ_minus_r = T(φ  -(r + fixed_low + fixed_up))
        abs_φ += abs(T(r + fixed_low + fixed_up))

        # Stop if φ-r ≈ 0
        if abs(φ_minus_r) < eps(T) * abs_φ
            flag = 0
            break
        end

        # Update bracket interval
        # Note that the current and previous φ may be incompatible,
        # since the latter was computed after fixing variables at the previous iteration.
        # Thus, we store φ_minus_r for the secant step.
        if φ_minus_r < T0
            lo_λ = λ
            lo_φ = φ
        else
            up_λ = λ
            up_φ = φ
        end

        # Stop if the bracket interval is too small
        if up_λ - lo_λ < eps(T) * max(abs(lo_λ), abs(up_λ))
            flag = 0
            break
        end

        if φ′ > T0
            δ = φ_minus_r / φ′
            old_λ = λ
            λ -= δ

            # Stop if no progress was achieved
            if (abs(δ) < eps(T)) || (old_λ == λ)
                flag = 0
                break
            end

            if (λ >= up_λ) || (λ <= lo_λ)
                # Newton step falls outside the bracket interval
                λ = secant_step(
                    lo_λ, 
                    up_λ, 
                    T(lo_φ - (r + fixed_low + fixed_up)), 
                    T(up_φ - (r + fixed_low + fixed_up))
                )
            end
        else
            # φ′ = 0, so take the closest breakpoint to continue
            λ = closest_breakpoint(P, λ, φ_minus_r, chunks)

            if isinf(λ)
                # There is no breakpoint, problem is infeasible
                flag = 2
                break
            end
        end

        # Try to fix variables and update RHS for the next iteration
        if φ_minus_r > T0
            fixed_low += altmapreduce(c -> fix_variables_l(P, x, c), .+, chunks; init=(T0))
        else
            fixed_up += altmapreduce(c -> fix_variables_u(P, x, c), .+, chunks; init=(T0))
        end
        chunks = compress_chunks(chunks)
    end

    return iter, flag
end

"""
    iter, flag = cqk!(sol, P; maxiters=100, numthreads=Threads.nthreads(), x0=[], chunks=FixedChunk[])

Parallel semismooth Newton method to solve the continuous quadratic knapsack
problem
```
min 0.5*x'*D*x - a'*t  s.t.  b'*x = r,  l <= x <= u
```
where `D` is a diagonal matrix with all diagonal elements positive and `P` is
a `CQKProblem` structure that contains the problem data.

The solution in written in `sol`, which must have the appropriate size and
type. `iter` returns the number of Newton steps used. `flag` indicates success
(`flag = 0`), invalid data (`flag = 1`), infeasible problem (`flag = 2`) or
failure (`flag = 3`).

It is possible to pre-allocate, for efficiency, the workspace using:

`chunks = initialize_chunks(n; numthreads=Threads.nthreads())`

Then, it can be used in subsequent executions of `cqk!`:

`iter, flag = cqk!(sol, P; chunks=chunks)`

In this case, `numthreads` in `cqk!` will be ignored and the value
used when creating the workspace will be used instead.
"""
function cqk!(
    sol::Vector{T},
    P::CQKProblem{T,V};
    maxiters=100,
    numthreads=Threads.nthreads(),
    chunks::Vector{FixedChunk}=FixedChunk[],
    x0::Vector{T}=T[]
) where {T<:AbstractFloat,V<:Vector{T}}
    if isempty(chunks)
        chunks = initialize_chunks(length(P.a); numthreads=numthreads)
    end
    iter, flag = cqk_newton(P, x0, sol, chunks, maxiters)
    return iter, flag
end

"""
    sol, iter, flag = cqk(P; maxiters=100, numthreads=Threads.nthreads(), x0=[], chunks=FixedChunk[])

Variation of `cqk!`, that allocates a new vector for the solution,
returning it.
"""
function cqk(
    P::CQKProblem{T,V};
    maxiters=100,
    numthreads=Threads.nthreads(),
    chunks::Vector{FixedChunk}=FixedChunk[],
    x0::Vector{T}=T[]
) where {T<:AbstractFloat,V<:Vector{T}}
    sol = similar(P.a)
    iter, flag = cqk!(sol, P; maxiters=maxiters, numthreads=numthreads, chunks=chunks, x0=x0)
    return sol, iter, flag
end
