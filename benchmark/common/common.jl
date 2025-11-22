using NewtonCQK

using Printf
using LinearAlgebra
using SparseArrays
using ArgParse
using BenchmarkTools
using OhMyThreads: OhMyThreads
using DataFrames
using JLD2

include("jld2_read.jl")

# Project path
projectpath = isfile("Project.toml") ? "./" : "../"

# Include third-party software
include(joinpath("../third_party", "condat", "condat_interface.jl"))
include(
    joinpath(
        "../third_party",
        "Parallel-Simplex-Projection",
        "src",
        "simplex_and_l1ball",
        "simplex_wrap.jl"
    )
)
include(
    joinpath(
        "../third_party",
        "Parallel-Simplex-Projection",
        "src",
        "simplex_and_l1ball",
        "l1ball_wrap.jl"
    )
)
include(
    joinpath(
        "../third_party", "quadratic_knapsack_source", "cqn_interface.jl"
    )
)

# Benchmark
function estimatetime(b)
    b.params.gctrial = true
    b.params.gcsample = false
    b.params.evals = 1
    b.params.samples = 10000
    b.params.seconds = 2.0
    samples = run(b)
    return minimum(samples.times)
end
