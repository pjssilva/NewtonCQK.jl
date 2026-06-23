function cplex_cqk!(
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}},
    inds::Vector{Cint};
    nthreads = 1,
    timelimit = 10.0
)
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "cplex_cqk.so").cplex_cqk(
        n::Csize_t,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        inds::Ptr{Cint},
        sol::Ptr{Cdouble},
        nthreads::Csize_t,
        timelimit::Cdouble
    )::Cint
    return max(res, 0), (res >= 0) ? :solved : :failed
end

function cplex_cqk(
    P::CQKProblem{Float64,Vector{Float64}};
    nthreads = 1,
    timelimit = 10.0
)
    n = length(P.a)
    sol = similar(P.a)
    inds = collect(Cint, 0:(n-1))
    iter, flag = cplex_cqk!(sol, P, inds; nthreads=nthreads, timelimit=timelimit)
    return sol, iter, flag
end
