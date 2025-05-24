############################################
# PROJECTION ONTO THE l1 BALL (CUDA VERSION)
# x; sum_i |x[i]| <= r
############################################

function cul1ball_solution_step(absy::T, y::T, λ::T) where {T<:AbstractFloat}
    return copysign(max(zero(T), absy + λ), y)
end

"""
    iter = cul1ball_proj!(sol, y; r=1.0, maxiters=100)

CUDA version of `l1ball_proj!`. `sol` and `x` must be `CuVector`s. The
projection is returned as the `CuVector` `sol`.
"""
function cul1ball_proj!(
    sol::CuVector{T}, y::CuVector{T}; r=one(T), maxiters=100, x0::CuVector{T}=CuVector{T}[]
) where {T<:AbstractFloat}
    absy = @~ abs.(y)
    if sum(absy) <= r
        sol .= y
        return 0
    end

    absx0 = @~ abs.(x0)
    λ, iter, solved = cusimplex_newton(absy, absx0, r, maxiters)

    if solved
        let λ = λ
            map!((absy, y) -> cul1ball_solution_step(absy, y, λ), sol, absy, y)
        end
        return iter
    else
        return min(-iter, -1)
    end
end

"""
    iter = cul1ball_proj!(y; r=1.0, maxiters=100)

replaces `y` by the projection.
"""
function cul1ball_proj!(
    y::CuVector{T}; r=one(T), maxiters=100, x0::CuVector{T}=CuVector{T}[]
) where {T<:AbstractFloat}
    return cul1ball_proj!(y, y; r=r, maxiters=maxiters, x0=x0)
end

"""
    sol, iter = cul1ball_proj!(y; r=1.0, maxiters=100)

CUDA version of `l1ball_proj`. `y` must be a `CuVector`. The projection
`sol` is also a `CuVector`.
"""
function cul1ball_proj(
    y::CuVector{T}; r=one(T), maxiters=100, x0::CuVector{T}=CuVector{T}[]
) where {T<:AbstractFloat}
    sol = similar(y)
    iter = cul1ball_proj!(sol, y; r=r, maxiters=maxiters, x0=x0)
    return sol, iter
end
