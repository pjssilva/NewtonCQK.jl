# Pre-allocate CPLEX model, configure it, define the linear constraint and
# bounds on variables, and returns a C pointer to the structure
function cplex_init(
    P::CQKProblem{Float64,Vector{Float64}};
    nthreads = 1,
    timelimit = 10.0
)
    n = length(P.a)
    cplex_pointer =
        @ccall joinpath(dirname(@__FILE__), "cplex_cqk.so").CPLEX_model_create(
        n::Csize_t,
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        nthreads::Cint,
        timelimit::Cdouble
        )::Ptr{Cvoid}
    return cplex_pointer
end

# Free pre-allocated CPLEX structure
function cplex_free(cplex_pointer)
    @ccall joinpath(dirname(@__FILE__), "cplex_cqk.so").CPLEX_model_free(
        cplex_pointer::Ptr{Cvoid})::Ptr{Cvoid}
end

# Solve a problem reusing the pre-allocated CPLEX structure
# Only the objective function is redefined, the rest remains unchanged.
# This function already consider the minus sign in the linear term of the
# objective function
function cplex_cqk!(
    cplex_pointer::Ptr{Cvoid},
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}}
)
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "cplex_cqk.so").cplex_cqk(
        cplex_pointer::Ptr{Cvoid},
        n::Csize_t,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        sol::Ptr{Cdouble}
    )::Cint
    return max(res, 0), (res >= 0) ? :solved : :failed
end

# Solve a problem allocating a new CPLEX structure
function cplex_cqk(
    P::CQKProblem{Float64,Vector{Float64}};
    nthreads = 1,
    timelimit = 10.0
)
    n = length(P.a)
    sol = similar(P.a)
    cplex_pointer = cplex_init(P, nthreads=nthreads, timelimit=timelimit)
    iter, flag = cplex_cqk!(cplex_pointer, sol, P)
    cplex_free(cplex_pointer)
    return sol, iter, flag
end
