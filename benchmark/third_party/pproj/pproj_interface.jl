function pproj_proj!(sol, x, r)
    n = length(x)
    libpath = @ccall joinpath(dirname(@__FILE__), "pproj_cqk.so").pproj_proj(
        x::Ptr{Cdouble}, sol::Ptr{Cdouble}, n::Csize_t, r::Cdouble
    )::Cvoid
    return nothing
end

function pproj_proj(x, r)
    sol = similar(x)
    pproj_proj!(sol, x, r)
    return sol
end

function pproj_cqk!(
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}}
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
        sol::Ptr{Cdouble}
    )::Cint
    return max(res, 0), (res >= 0) ? :solved : :failed
end

function pproj_cqk(P::CQKProblem{Float64,Vector{Float64}})
    n = length(P.a)
    sol = similar(P.a)
    iter, flag = pproj_cqk!(sol, P)
    return sol, iter, flag
end
