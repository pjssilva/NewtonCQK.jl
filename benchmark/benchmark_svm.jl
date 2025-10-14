using NewtonCQK

using Printf
using Random: Random
Random.seed!(0)
using LinearAlgebra
using BenchmarkTools
using ArgParse
using OhMyThreads: OhMyThreads
using DataFrames
using JLD2
using Distances
using UCIData

function estimatetime(b)
    b.params.gctrial = true
    b.params.gcsample = false
    b.params.evals = 1
    b.params.samples = 10000
    b.params.seconds = 2.0
    samples = run(b)
    return minimum(samples.times)
end

function get_parameters()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--continue"
        arg_type = Bool
        default = false
        help = "continue previous tests?"
    end
    return parse_args(s)
end

include("spg.jl")

function svm(
    instance, Z, w, nthreads, results;
    sigma = 5.0,
    C = 1.0
)
    @assert sigma > 0.0 throw(ArgumentError("sigma must be positive"))
    @assert C > 0.0 throw(ArgumentError("C must be positive"))
    @assert !isempty(instance) throw(ArgumentError("instance must be provided"))

    output = "results_svm.jld2"

    n = size(Z,2)

    # CQK for subproblems
    P = CQKProblem(ones(n), zeros(n), w, 0.0, zeros(n), fill(C,n))

    # Kernel
    frac = 0.5 / sigma^2
    K(zi,zj) = exp(frac * sqeuclidean(zi,zj))

    # "Lazy" Hessian
    H(Z) = @inbounds @views [ K(Z[:,i],Z[:,j]) for i=1:n, j=1:n ]

    # Objective function
    function f(x)
        return 0.5 * (x' * H(Z) * x) - sum(x)
    end

    # Gradient of objective function
    function g!(g, x)
        g .= (H(Z) * x) .- 1.0
    end

    # Direction (solve particular CQK)
    function d!(d, x, lambda, g)
        @. P.a = x - lambda * g
        cqk!(d, P)
        d .-= x
    end

    # Projected gradient supnorm
    tmp = similar(P.a)
    function pg_supnorm(x, g)
        d!(tmp, x, 1.0, g)
        return norm(tmp, Inf)
    end

    # Callback function
    # We do not store every iterate produced by SPG because this can consume too
    # memory. Instead, we perform the benchmarks here and store only the result.
    # P is relative to x, which must be updated by d! before. For benchmarking,
    # we start at x0 = xprev = "previous x". In the first outer iteration, x0 is
    # undefined.
    xprev = []
    sol = similar(P.a)
    function b_callback(x, iter)
        # CQK
        chunks = initialize_chunks(length(sol); nchunks=nthreads)
        b = @benchmarkable cqk!($sol, $P, chunks=($chunks), x0=($xprev))
        time = estimatetime(b)
        iter, flag = cqk(P, nchunks=nthreads, x0=(xprev))[2:3]
        infeas = abs(dot(P.b, sol) - P.r)
        push!(results, [instance, n, "cqk", nthreads, iter, flag, time, infeas])

        # CMS_CQN
        if nthreads == 1
            b = @benchmarkable cms_cqn!($sol, $P)
            time = estimatetime(b)
            iter, flag = cms_cqn(P, x0=(xprev))[2:3]
            infeas = abs(dot(P.b, sol) - P.r)
            push!(results, [instance, n, "cqk", 1, iter, flag, time, infeas])
        end

        # Update xprev for the next round
        xprev .= x
    end

    # --------
    # CALL SPG
    # --------
    spg(n, f, g!, d!, pg_supnorm, l = 0.0, u = C, callback = b_callback)

    # Save results
    jldsave(output; results)

    return
end

function executed(results, instance, nthreads)
    if !isempty(
        results[
            (results.Instance .== instance) .& (results.threads .== nthreads),
            :
        ]
    )
        @printf(
            "%12s  %3d  already executed. Skipping...\n",
            instance, nthreads
        )
        return true
    end
    return false
end

function main(args)
    nthreads = Threads.nthreads()

    # Get command line parameters
    opts = get_parameters()

    # Results
    results = DataFrame(
        [
            "Instance" => String[]
            "n" => Int64[]
            "Algorithm" => String[]
            "threads" => Int64[]
            "iter" => Int64[]
            "st" => []
            "time" => Float64[]
            "infeas" => Float64[]
        ]
    )
    if opts["continue"] && isfile(output)
        jld2file = jldopen(output, "r")
        results = read(jld2file, "results")
        close(jld2file)
    end

    # IRIS
    if !executed(results, "iris", nthreads)
        println("\nDataset: iris\n")
        data = UCIData.dataset("iris")
        Z = Matrix(data[1:100, 2:5])'
        w = ones(100)
        w[1:50] .= -1.0

        svm("iris", Z, w, nthreads, results; sigma = 5.0, C = 1.0)
    end

    return 0
end

# Run main if non-iteractive
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
