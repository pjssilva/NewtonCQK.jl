##############################################
# PROJECTION y ONTO THE SIMPLEX (CUDA VERSION)
# x; sum_i x[i] == r,  x >= 0
##############################################

# To initialize λ when x0 is provided. It is assumed that x0 >= 0.
function cusimplex_init(y::T, x0::T, r::T) where {T<:AbstractFloat}
    return x0 > zero(T) ? (y, one(T)) : (zero(T), zero(T))
end

function cusimplex_phi_step(y::T, λ::T) where {T<:AbstractFloat}
    new_x = y + λ
    return new_x >= zero(T) ? (new_x, one(T)) : (zero(T), zero(T))
end

function cusimplex_newton(
    y::CuVector{T}, x0::CuVector{T}, r, maxiters
) where {T<:AbstractFloat}
    T0 = zero(T)

    # initialize λ
    if isempty(x0)
        λ = (r - sum(y)) / length(y)

        if abs(λ) < eps(T)
            # y is the solution!
            return T0, 0, true
        end
    else
        sumy, len = let r = r
            mapreduce((y, x0) -> cusimplex_init(y, x0, r), .+, y, x0; init=(T0))
        end

        if len > 0
            λ = (r - sumy) / len
        else
            λ = -maximum(y)
        end
    end

    # Newton loop
    iter = 0
    solved = false
    while (iter < maxiters)
        φ, φ′ = let λ = λ
            mapreduce(y -> cusimplex_phi_step(y, λ), .+, y; init=(T0, T0))
        end
        δ = (φ - r) / φ′
        old_λ = λ
        λ -= δ
        if (δ < eps(T)) || (old_λ == λ)
            # If δ is too small or it does not modify λ, stop. In both cases, φ-r ≈ 0.
            solved = true
            break
        end
        iter += 1
    end

    return λ, iter, solved
end

"""
    iter = cusimplex_proj!(sol, y; r=1.0, maxiters=100, x0=CuVector[])

CUDA version of `simplex_proj!`. `sol`, `x` and `x0` must be `CuVector`s.
"""
function cusimplex_proj!(
    sol::CuVector{T}, y::CuVector{T}; r=one(T), maxiters=100, x0::CuVector{T}=CuVector{T}[]
) where {T<:AbstractFloat}
    λ, iter, solved = cusimplex_newton(y, r, maxiters)

    if solved
        @. sol = max(zero(T), y + λ)
        return iter
    else
        return min(-iter, -1)
    end
end

"""
    iter = cusimplex_proj!(y; r=1.0, maxiters=100, x0=CuVector[])

A CUDA version of `simplex_proj!` that returns the solution in `y` itself.
"""
function cusimplex_proj!(
    y::CuVector{T}; r=one(T), maxiters=100, x0::CuVector{T}=CuVector{T}[]
) where {T<:AbstractFloat}
    return cusimplex_proj!(y, y; r=r, maxiters=maxiters, x0=x0)
end

"""
    sol, iter = cusimplex_proj(y; r=1.0, maxiters=100, x0=CuVector[])

CUDA version of `simplex_proj`. `y` and `x0` must be `CuVector`'s.
The projection `sol` is also a `CuVector`.
"""
function cusimplex_proj(
    y::CuVector{T}; r=one(T), maxiters=100, x0::CuVector{T}=CuVector{T}[]
) where {T<:AbstractFloat}
    sol = similar(y)
    iter = cusimplex_proj!(sol, y; r=r, maxiters=maxiters, x0=x0)
    return sol, iter
end
