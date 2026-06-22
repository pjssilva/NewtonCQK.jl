function pproj_proj!(
    sol,
    x,
    Ap::Vector{Cint},
    Ai::Vector{Cint},
    zrs,
    ons;
    tol = 1e-6
)
    n = length(x)
    res = @ccall joinpath(dirname(@__FILE__), "pproj_cqk.so").pproj_proj(
        n::Csize_t,
        x::Ptr{Cdouble},
        Ap::Ptr{Cint},
        Ai::Ptr{Cint},
        zrs::Ptr{Cdouble},
        ons::Ptr{Cdouble},
        sol::Ptr{Cdouble},
        tol::Cdouble
    )::Cint
    return (res == 0) ? :solved : :failed
end

function pproj_proj(x; tol = 1e-6)
    n = length(x)
    sol = similar(x)
    Ap = collect(Cint, 0:n)
    Ai = zeros(Cint, n)
    zrs = zeros(Cdouble, n)
    ons = ones(Cdouble, n)
    flag = pproj_proj!(sol, x, Ap, Ai, zrs, ons; tol=tol)
    return sol, flag
end

# NOTE: P.a, P.b, P.low, P.up must be scaled to represent the projection
# of D^{-1/2}a. In turn, P.d must be unchanged. Function CPUtoPPROJ in
# convert.jl do this.
function pproj_cqk!(
    sol::Vector{Float64},
    P::CQKProblem{Float64,Vector{Float64}},
    Ap::Vector{Cint}, Ai::Vector{Cint};
    tol = 1e-6
)
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "pproj_cqk.so").pproj_cqk(
        n::Csize_t,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        Ap::Ptr{Cint},
        Ai::Ptr{Cint},
        sol::Ptr{Cdouble},
        tol::Cdouble
    )::Cint
    return (res >= 0) ? :solved : :failed
end

# NOTE: P.a, P.b, P.low, P.up must be scaled to represent the projection
# of D^{-1/2}a. In turn, P.d must be unchanged. Function CPUtoPPROJ in
# convert.jl do this.
function pproj_cqk(P::CQKProblem{Float64,Vector{Float64}}; tol = 1e-6)
    n = length(P.a)
    sol = similar(P.a)
    Ap = collect(Cint, 0:n)
    Ai = zeros(Cint, n)
    flag = pproj_cqk!(sol, P, Ap, Ai; tol=tol)
    return sol, flag
end
