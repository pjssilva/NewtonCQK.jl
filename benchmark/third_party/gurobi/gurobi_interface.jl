# Pre-allocate Gurobi model, configure it, define the linear constraint and
# bounds on variables, and returns a C pointer to the structure
function gurobi_init(
    P::CQKProblem{Float64,Vector{Float64}};
    nthreads = 1,
    timelimit = 10.0
)
    n = length(P.a)
    gurobi_pointer =
        @ccall joinpath(dirname(@__FILE__), "gurobi_cqk.so").GUROBI_model_create(
        n::Csize_t,
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        nthreads::Cint,
        timelimit::Cdouble
        )::Ptr{Cvoid}
    return gurobi_pointer
end

# Free pre-allocated Gurobi structure
function gurobi_free(gurobi_pointer)
    @ccall joinpath(dirname(@__FILE__), "gurobi_cqk.so").GUROBI_model_free(
        gurobi_pointer::Ptr{Cvoid})::Ptr{Cvoid}
end

# Solve a problem reusing the pre-allocated Gurobi structure
# Only the objective function is redefined, the rest remains unchanged.
# This function already consider the minus sign in the linear term of the
# objective function
function gurobi_cqk!(
    gurobi_pointer::Ptr{Cvoid},
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}}
)
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "gurobi_cqk.so").gurobi_cqk(
        gurobi_pointer::Ptr{Cvoid},
        n::Csize_t,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        sol::Ptr{Cdouble}
    )::Cint
    return max(res, 0), (res >= 0) ? :solved : :failed
end

# Solve a problem allocating a new Gurobi structure
function gurobi_cqk(
    P::CQKProblem{Float64,Vector{Float64}};
    nthreads = 1,
    timelimit = 10.0
)
    n = length(P.a)
    sol = similar(P.a)
    gurobi_pointer = gurobi_init(P, nthreads=nthreads, timelimit=timelimit)
    iter, flag = gurobi_cqk!(gurobi_pointer, sol, P)
    return sol, iter, flag
end
