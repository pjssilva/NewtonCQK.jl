function hexaly_cqk!(
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}};
    x0 = Float64[],
    nthreads = 1,
    timelimit = 10
)
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "hexaly_cqk.so").hexaly_cqk(
        n::Csize_t,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        sol::Ptr{Cdouble},
        nthreads::Csize_t,
        timelimit::Csize_t
    )::Cint
    return max(res, 0), (res >= 0) ? :solved : :failed
end

function hexaly_cqk(
    P::CQKProblem{Float64,Vector{Float64}};
    x0 = Float64[],
    nthreads = 1,
    timelimit = 10
)
    sol = similar(P.a)
    iter, flag = hexaly_cqk!(
        sol, P, x0=x0, nthreads=nthreads, timelimit=timelimit
    )
    return sol, iter, flag
end
