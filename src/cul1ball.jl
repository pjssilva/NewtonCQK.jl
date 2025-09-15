############################################
# PROJECTION ONTO THE l1 BALL (CUDA VERSION)
# x; sum_i |x[i]| <= r
############################################

function cul1ball_solution_step(y::T, λ::T) where {T<:AbstractFloat}
    return copysign(max(0, abs(y) + λ), y)
end

"""
    iter, flag = cul1ball_proj!(sol, y; r=1.0, maxiters=100)

CUDA version of `l1ball_proj!`. `sol` and `x` must be `CuVector`s. The
projection is returned as the `CuVector` `sol`.
"""
function l1ball_proj!(
    sol::CuVector{T}, y::CuVector{T}; r=one(T), maxiters=100, x0::CuVector{T}=CuVector{T}(undef, 0)
) where {T<:AbstractFloat}
    absy = @~ abs.(y)
    if sum(absy) <= r
        sol .= y
        return 0, :solved
    end

    absx0 = @~ abs.(x0)
    λ, iter, flag = cusimplex_newton(absy, absx0, r, maxiters)

    let λ = λ
        map!(y -> cul1ball_solution_step(y, λ), sol, y)
    end
    return iter, flag
end

"""
    iter, flag = cul1ball_proj!(y; r=1.0, maxiters=100)

replaces `y` by the projection.
"""
function l1ball_proj!(
    y::CuVector{T}; r=one(T), maxiters=100, x0::CuVector{T}=CuVector{T}(undef, 0)
) where {T<:AbstractFloat}
    return l1ball_proj!(y, y; r=r, maxiters=maxiters, x0=x0)
end

"""
    sol, iter, flag = cul1ball_proj!(y; r=1.0, maxiters=100)

CUDA version of `l1ball_proj`. `y` must be a `CuVector`. The projection
`sol` is also a `CuVector`.
"""
function l1ball_proj(
    y::CuVector{T}; r=one(T), maxiters=100, x0::CuVector{T}=CuVector{T}(undef, 0)
) where {T<:AbstractFloat}
    sol = similar(y)
    iter, flag = l1ball_proj!(sol, y; r=r, maxiters=maxiters, x0=x0)
    return sol, iter, flag
end
