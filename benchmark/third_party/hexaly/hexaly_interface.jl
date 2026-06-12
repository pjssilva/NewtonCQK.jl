function hexaly_cqk!(
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}};
    x0 = Float64[]
)
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "lib", "hexaly_cqk.so").hexaly_cqk(
        n::Cint,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        sol::Ptr{Cdouble}
    )::Cint
    return max(res, 0), (res >= 0) ? :solved : :max_iter
end

function hexaly_cqk(P::CQKProblem{Float64,Vector{Float64}}; x0 = Float64[])
    n = length(P.a)
    sol = similar(P.a)
    iter, flag = hexaly_cqk!(sol, P, x0=x0)
    return sol, iter, flag
end
