function cms_cqn!(sol::Vector{Float64}, P::CQKProblem{Float64,Vector{Float64}})
    n = length(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "lib", "newtonproj.so").newton_cqn(
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

function cms_cqn(P::CQKProblem{Float64,Vector{Float64}})
    n = length(P.a)
    sol = similar(P.a)
    res = @ccall joinpath(dirname(@__FILE__), "lib", "newtonproj.so").newton_cqn(
        n::Cint,
        P.d::Ptr{Cdouble},
        P.a::Ptr{Cdouble},
        P.b::Ptr{Cdouble},
        P.r::Cdouble,
        P.l::Ptr{Cdouble},
        P.u::Ptr{Cdouble},
        sol::Ptr{Cdouble}
    )::Cint
    return sol, max(res, 0), (res >= 0) ? :solved : :max_iter
end

#############################################

# function cms_proj(
#     x::Vector{Float64},
#     b::Vector{Float64},
#     r::Float64,
#     low::Vector{Float64},
#     up::Vector{Float64},
# )
#     n = length(x)
#     sol = similar(x)
#     @ccall datadir(
#         "third_party", "quadratic_knapsack_source", "lib", "newtonproj.so"
#     ).newton_projection_2(
#         n::Cint,
#         x::Ptr{Cdouble},
#         b::Ptr{Cdouble},
#         r::Cdouble,
#         low::Ptr{Cdouble},
#         up::Ptr{Cdouble},
#         sol::Ptr{Cdouble},
#     )::Cvoid
#     return sol
# end
#
# function cms_proj!(
#     sol::Vector{Float64},
#     x::Vector{Float64},
#     b::Vector{Float64},
#     r::Float64,
#     low::Vector{Float64},
#     up::Vector{Float64},
# )
#     n = length(x)
#     @ccall datadir(
#         "third_party", "quadratic_knapsack_source", "lib", "newtonproj.so"
#     ).newton_projection_2(
#         n::Cint,
#         x::Ptr{Cdouble},
#         b::Ptr{Cdouble},
#         r::Cdouble,
#         low::Ptr{Cdouble},
#         up::Ptr{Cdouble},
#         sol::Ptr{Cdouble},
#     )::Cvoid
#     return
# end
#
# function simplex_cms_proj(x::Vector{Float64})
#     r = 1.0
#     n = length(x)
#     b = ones(n)
#     low = zeros(n)
#     up = fill(Inf, n)
#     return cms_proj(x, b, r, low, up)
# end
