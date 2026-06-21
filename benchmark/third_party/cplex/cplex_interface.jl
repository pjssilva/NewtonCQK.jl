function cplex_cqk!(
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}};
    x0 = Float64[],
    nthreads = 1,
    method = 0,
    # 0 Automatic: let CPLEX choose; default
    # 1 Use the primal simplex optimizer.
    # 2 Use the dual simplex optimizer.
    # 3 Use the network optimizer.
    # 4 Use the barrier optimizer.
    # 6 Use the concurrent optimizer.
    timelimit = 10.0
)
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "cplex_cqk.so").cplex_cqk(
        n::Cint,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        sol::Ptr{Cdouble},
        nthreads::Cint,
        method::Cint,
        timelimit::Cdouble
    )::Cint
    return max(res, 0), (res >= 0) ? :solved : :failed
end

function cplex_cqk(
    P::CQKProblem{Float64,Vector{Float64}};
    x0 = Float64[],
    nthreads = 1,
    method = 0,
    timelimit = 10.0
)
    sol = similar(P.a)
    iter, flag = cplex_cqk!(
        sol, P, x0=x0, nthreads=nthreads, method=method, timelimit=timelimit
    )
    return sol, iter, flag
end
