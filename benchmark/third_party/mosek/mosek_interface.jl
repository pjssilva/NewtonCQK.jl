# Pre-allocate MOSEK model, configure it, define the linear constraint and
# bounds on variables, and returns a C pointer to the structure
function mosek_init(
    P::CQKProblem{Float64,Vector{Float64}};
    nthreads = 1,
    timelimit = 10.0
)
    n = length(P.a)
    mosek_pointer =
        @ccall joinpath(dirname(@__FILE__), "mosek_cqk.so").MOSEK_model_create(
        n::Csize_t,
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        nthreads::Cint,
        timelimit::Cdouble
        )::Ptr{Cvoid}
    return mosek_pointer
end

# Free pre-allocated MOSEK structure
function mosek_free(mosek_pointer)
    @ccall joinpath(dirname(@__FILE__), "mosek_cqk.so").MOSEK_model_free(
        mosek_pointer::Ptr{Cvoid})::Ptr{Cvoid}
end

# Solve a problem reusing the pre-allocated Gurobi structure
# Only the objective function is redefined, the rest remains unchanged.
# This function already consider the minus sign in the linear term of the
# objective function
function mosek_cqk!(
    mosek_pointer::Ptr{Cvoid},
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}}
)
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "mosek_cqk.so").mosek_cqk(
        mosek_pointer::Ptr{Cvoid},
        n::Csize_t,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        sol::Ptr{Cdouble}
    )::Cint
    return max(res, 0), (res >= 0) ? :solved : :failed
end

# Solve a problem allocating a new MOSEK structure
function mosek_cqk(
    P::CQKProblem{Float64,Vector{Float64}};
    nthreads = 1,
    timelimit = 10.0
)
    sol = similar(P.a)
    mosek_pointer = mosek_init(P, nthreads=nthreads, timelimit=timelimit)
    iter, flag = mosek_cqk!(mosek_pointer, sol, P)
    mosek_free(mosek_pointer)
    return sol, iter, flag
end
