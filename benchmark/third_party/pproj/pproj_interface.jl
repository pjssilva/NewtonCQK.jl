function pproj_proj!(sol, x; tol = 1e-9)
    n = length(x)
    res = @ccall joinpath(dirname(@__FILE__), "pproj_cqk.so").pproj_proj(
        n::Csize_t, x::Ptr{Cdouble}, sol::Ptr{Cdouble}, tol::Cdouble
    )::Cvoid
    return (res == 0) ? :solved : :failed
end

function pproj_proj(x; tol = 1e-9)
    sol = similar(x)
    res = pproj_proj!(sol, x; tol=tol)
    return sol, (res == 0) ? :solved : :failed
end

function pproj_cqk!(
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}};
    tol = 1e-9
)
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "pproj_cqk.so").pproj_cqk(
        n::Cint,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        sol::Ptr{Cdouble},
        tol::Cdouble
    )::Cint
    return max(res, 0), (res >= 0) ? :solved : :failed
end

function pproj_cqk(P::CQKProblem{Float64,Vector{Float64}}; tol = 1e-9)
    n = length(P.a)
    sol = similar(P.a)
    iter, flag = pproj_cqk!(sol, P, tol=tol)
    return sol, iter, flag
end
