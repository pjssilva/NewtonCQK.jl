using DrWatson
@quickactivate "ParallelNewtonCQK"
using NewtonCQK

using LinearAlgebra
using BenchmarkTools
using ProximalOperators

# Include the third party solvers
include(joinpath("third_party", "condat", "condat_interface.jl"))
include(
    joinpath(
        "third_party",
        "Parallel-Simplex-Projection",
        "src",
        "simplex_and_l1ball",
        "simplex_wrap.jl",
    ),
)
include(joinpath("third_party", "quadratic_knapsack_source", "cqn_interface.jl"))

# Function to generate a random problem
function random_x(n)
    fillin = max(1, Int(floor(rand()^2 * n)))
    inds = rand(1:n, fillin)
    x = zeros(n)
    x[inds] = rand(fillin)
    return x
end

# Generate a random problem
n = 20_000_000
x = random_x(n)

# Solve with various solvers
pcondat = condat_proj(x)
paltcondat = condat_s(x)
psp, _ = simplex_proj(x)

# Compare RHS
@show sum(pcondat)
@show sum(paltcondat)
@show sum(psp)

# Calculate distances
@show norm(pcondat .- paltcondat)
@show norm(pcondat .- psp)
