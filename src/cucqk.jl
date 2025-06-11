##############################################
# SOLVE CONTINUOUS QUADRATIC KNAPSACK PROBLEMS (CUDA VERSION)
#
# min 0.5*x'*D*x - a'*t  s.t.  b'*x = r,  l <= x <= u
#
# It is supposed that b > 0 and D = Diagonal(d) with d > 0.
##############################################

# Initialize λ when x0 is not provided
function cucqk_init(d::T, a::T, b::T, l::T, u::T) where {T<:AbstractFloat}
    if (d <= zero(T)) || (b <= zero(T)) || (l > u)
        # Invalid data
        return zero(T), T(Inf)
    else
        return (b * a) / d, (b * b) / d
    end
end

# Initialize λ when x0 is provided
function cucqk_init(d::T, a::T, b::T, l::T, u::T, x0::T) where {T<:AbstractFloat}
    if (d <= zero(T)) || (b <= zero(T)) || (l > u)
        # Invalid data
        return zero(T), T(Inf), zero(T)
    else
        if x0 <= l
            return zero(T), zero(T), -(b * l)
        elseif x0 >= u
            return zero(T), zero(T), -(b * u)
        else
            return (b * a) / d, (b * b) / d, zero(T)
        end
    end
end

@inline function cucqk_phi_step(d::T, a::T, b::T, l::T, u::T, λ::T) where {T<:AbstractFloat}
    new_x = (b * λ + a) / d
    if new_x >= l
        if new_x <= u
            # As in the implementation of Comineti, Mascarenhas and Silva,
            # we do not differ from negative or positive derivatives
            phi = b * new_x
            return phi, b * (b / d), abs(phi)
        else
            phi = b * u
            return phi, zero(T), abs(phi)
        end
    else
        phi = b * l
        return phi, zero(T), abs(phi)
    end
end

function cubreakpoint_to_the_right(d::T, a::T, b::T, u::T, λ::T) where {T<:AbstractFloat}
    bkp = (d * u - a) / b
    return bkp > λ ? bkp : T(Inf)
end

function cubreakpoint_to_the_left(d::T, a::T, b::T, l::T, λ::T) where {T<:AbstractFloat}
    bkp = (d * l - a) / b
    return bkp < λ ? bkp : T(-Inf)
end

function cucqk_solution(d::T, a::T, b::T, l::T, u::T, λ::T) where {T<:AbstractFloat}
    new_x = (b * λ + a) / d
    if new_x > u
        return u
    elseif new_x < l
        return l
    else
        return new_x
    end
end

function cucqk_newton(
    P::CQKProblem{T,V}, x0::CuVector{T}, x::CuVector{T}, maxiters
) where {T<:AbstractFloat,V<:CuVector{T}}
    T0 = zero(T)
    lo_λ = T(-Inf)
    up_λ = T(Inf)
    lo_φ = lo_λ
    up_φ = up_λ

    # λ initialization
    if isempty(x0)
        s, q = mapreduce(cucqk_init, .+, P.d, P.a, P.b, P.l, P.u; init=(T0, T0))
        λ = (P.r - s) / q
    else
        s, q, r_aux = mapreduce(
            cucqk_init, .+, P.d, P.a, P.b, P.l, P.u, x0; init=(T0, T0, T0)
        )
        if q == T0
            s, q = mapreduce(
                cucqk_init, .+, P.d, P.a, P.b, P.l, P.u; init=(T0, T0)
            )
            λ = (P.r - s) / q
        else
            λ = (P.r + r_aux - s) / q
        end
    end

    # q is Inf if data is inconsistent or an infeasibility was identified
    if isinf(q)
        return T0, 0, 1
    end

    flag = 3

    # Newton loop
    iter = 0
    while (iter < maxiters)
        # Compute φ-r and φ'
        φ_minus_r, φ′, abs_φ = let λ = λ
            mapreduce(
                (d, a, b, l, u) -> cucqk_phi_step(d, a, b, l, u, λ),
                .+,
                P.d,
                P.a,
                P.b,
                P.l,
                P.u;
                init=(T0, T0, T0)
            )
        end
        φ_minus_r -= P.r

        # Stop if φ-r ≈ 0
        if abs(φ_minus_r) < eps(T) * (abs(P.r) + abs_φ)
            flag = 0
            break
        end

        # Update bracket interval
        # The current and previous φ are compatible here as no fixing variable is performed.
        # However, to maintain compatibility with CPU version, we store φ_minus_r.
        if φ_minus_r < T0
            lo_λ = λ
            lo_φ = φ_minus_r
        else
            up_λ = λ
            up_φ = φ_minus_r
        end

        # Stop if the bracket interval is too small
        if up_λ - lo_λ < eps(T) * max(abs(lo_λ), abs(up_λ))
            flag = 0
            break
        end

        iter += 1

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
                λ = secant_step(lo_λ, up_λ, lo_φ, up_φ)
            end
        else
            if φ_minus_r < T0
                λ = let λ = λ
                    mapreduce(
                        (d, a, b, u) -> cubreakpoint_to_the_right(d, a, b, u, λ),
                        min,
                        P.d,
                        P.a,
                        P.b,
                        P.u;
                        init=(T0)
                    )
                end
            else
                λ = let λ = λ
                    mapreduce(
                        (d, a, b, l) -> cubreakpoint_to_the_left(d, a, b, l, λ),
                        max,
                        P.d,
                        P.a,
                        P.b,
                        P.l;
                        init=(T0)
                    )
                end
            end

            if isinf(λ)
                # There is no breakpoint, problem is infeasible
                flag = 2
                break
            end
        end
    end

    return λ, iter, flag
end

"""
    iter, flag = cqk!(sol, P; maxiters=100, x0=[])

CUDA version of `cqk!`. `sol` and `x0` must be `CuVector`'s.
"""
function cqk!(
    sol::CuVector{T}, P::CQKProblem{T,V}; maxiters=100, x0::CuVector{T}=CuVector{T}(undef, 0)
) where {T<:AbstractFloat,V<:CuVector{T}}
    λ, iter, flag = cucqk_newton(P, x0, sol, maxiters)

    if flag == 0
        let λ = λ
            map!(
                (d, a, b, l, u) -> cucqk_solution(d, a, b, l, u, λ),
                sol,
                P.d,
                P.a,
                P.b,
                P.l,
                P.u
            )
        end
    end
    return iter, flag
end

"""
    sol, iter, flag = cqk(P; maxiters=100, x0=[])

CUDA version of `cqk`. `x0` must be a `CuVector`.
"""
function cqk(
    P::CQKProblem{T,V}; maxiters=100, x0::CuVector{T}=CuVector{T}(undef, 0)
) where {T<:AbstractFloat,V<:CuVector{T}}
    sol = similar(P.a)
    iter, flag = cucqk!(sol, P; maxiters=maxiters, x0=x0)
    return sol, iter, flag
end
