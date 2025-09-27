using NewtonCQK

using Printf
using Random: Random
using LinearAlgebra
using BenchmarkTools
using ArgParse
using OhMyThreads: OhMyThreads
using DataFrames
using CSV
using Distributions
using JLD2
using CUDA
using SparseArrays

using ThreadPinning
pinthreads(:cores)

function get_parameters()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--cuda"
        arg_type = Int
        default = 0
        help = "benchcmark a CUDA GPU (32 for 32bit only or 64 for 32 and 64 bits)"
        "--nreps"
        arg_type = Int
        default = 1
        help = "number of problems to solve"
        "--continue"
        arg_type = Bool
        default = false
        help = "continue previous tests?"
    end
    return parse_args(s)
end

# Functions to convert data
include("convert.jl")

##############################
# Structure for instances
struct INSTANCE
    name::String            # unique name
    n::Int64                # number of variables
    generate::Function      # function to generates the instance
end

# Structure for methods
struct METHOD
    name::String            # unique name
    convert::Function       # function to convert data
    time::Function          # function that returns the CPU time
    it_st::Function         # function that returns #iterations and status
    infeas::Function        # function that returns the infeasibility
    extra_info::Function    # extra info; for GPU algorithms, may be used to
    # compute the relative error of the solution
end

# Structure that match methods and instances
struct TEST
    title::String
    instances::Vector{INSTANCE}   # vector of instances
    methods::Vector{METHOD}       # vector of methods to be applied to each instance
end

"""
Computes the maximum problem size that fits both RAM and GPU memory.
"""
function maxsize(unitsize64, usecuda)
    ram_size = Int(Sys.total_memory())
    gpu_ram = usecuda > 0 ? CUDA.memory_info()[2] : Inf
    half_mem = min(ram_size, gpu_ram)
    return half_mem / unitsize64
end
USECUDA = get_parameters()["cuda"]

# Define instances and methods
include("tests_cqk.jl")
include("tests_simplex.jl")
include("tests_l1ball.jl")

# Assign instances to each method class
TESTS = TEST[]
push!(TESTS, TEST("CQK", CQK_INSTANCES, CQK_METHODS))
push!(TESTS, TEST("Simplex", SIMPLEX_INSTANCES, SIMPLEX_METHODS))
push!(TESTS, TEST("l1 ball", SIMPLEX_INSTANCES, l1BALL_METHODS))
##############################

include(joinpath("third_party", "condat", "condat_interface.jl"))
include(
    joinpath(
        "third_party",
        "Parallel-Simplex-Projection",
        "src",
        "simplex_and_l1ball",
        "simplex_wrap.jl"
    )
)
include(
    joinpath(
        "third_party",
        "Parallel-Simplex-Projection",
        "src",
        "simplex_and_l1ball",
        "l1ball_wrap.jl"
    )
)
include(joinpath("third_party", "quadratic_knapsack_source", "cqn_interface.jl"))

############################
# Benchmark
############################

# CPU time
function estimatetime(b)
    b.params.gctrial = false
    b.params.evals = 1
    b.params.samples = 1#00
    b.params.seconds = 10.0
    samples = run(b)
    return minimum(samples.times)
end

# Relative difference between FP64 and alternative solutions
function reldiff_sol(P, cpualg, altalg)
    alt_sol = altalg(P)[1]
    cpu_P = GPUtoCPU(P, Float64)
    cpu_sol = cpualg(cpu_P)[1]
    return norm(Vector(alt_sol) - cpu_sol) / norm(cpu_sol)
end

reldiff_sol(P, alg) = reldiff_sol(P, alg, alg)

# Benchmark for Parallel Condat
function b_Pcondat(P, serial_alg, parallel_alg, nthreads)
    if nthreads == 1
        b = @benchmarkable $serial_alg($P, 1.0)
    else
        b = @benchmarkable $parallel_alg($P, 1.0, $nthreads, 0.001)
    end
    time = estimatetime(b)
    return time
end

# Benchmark for Condat simplex projection (Condat's C code)
function b_Ccondat(P, nthreads)
    if nthreads == 1
        sol = similar(P)
        b = @benchmarkable condat_proj!($sol, $P)
        time = estimatetime(b)
        return time
    else
        # Condat's code is not parallel
        return Inf
    end
end

# Benchmark for CQN
function b_cms_cqn(P, nthreads)
    if nthreads == 1
        sol = similar(P.a)
        b = @benchmarkable cms_cqn!($sol, $P)
        time = estimatetime(b)
        return time
    else
        # CQN is not parallel
        return Inf
    end
end

# Benchmark of our Newton algorithms
function ref_obj(P::CQKProblem)
    return P.a
end
function ref_obj(P::Union{Vector,CuVector})
    return P
end

# Dense solution
function b_dense(
    P::Union{CQKProblem{T,V},V}, alg, nthreads; x0=T[]
) where {T<:AbstractFloat,V<:Vector{T}}
    sol = similar(ref_obj(P))
    chunks = initialize_chunks(length(sol); nchunks=nthreads)
    b = @benchmarkable $alg($sol, $P, chunks=($chunks), x0=($x0))
    time = estimatetime(b)
    return time
end

# Sparse solution
function b_sparse(P::Vector{T}, alg, nthreads; x0=T[]) where {T<:AbstractFloat}
    chunks = initialize_chunks(length(P); nchunks=nthreads)
    b = @benchmarkable $alg($P, chunks=($chunks), x0=($x0))
    time = estimatetime(b)
    return time
end

# CUDA
function b_cuda(
    P::Union{CQKProblem{T,V},V}, alg, nthreads; x0=CuVector{T}(undef, 0)
) where {T<:AbstractFloat,V<:CuVector{T}}
    if nthreads == 1
        unpinthreads()
        sol = similar(ref_obj(P))
        b = @benchmarkable CUDA.@sync $alg($sol, $P, x0=($x0))
        time = estimatetime(b)
        pinthreads(:cores)
        return time
    else
        return Inf
    end
end

############################

# Print time
function fnanosec(ns::Number)
    if ns < 1_000
        return @sprintf("%7.4g ns", ns)
    elseif ns < 1_000_000
        return @sprintf("%7.4g μs", ns / 1_000)
    elseif ns < 1_000_000_000
        return @sprintf("%7.4g ms", ns / 1_000_000)
    else
        return @sprintf("%7.4g  s", ns / 1_000_000_000)
    end
end

function print_test_header(title)
    @printf("\n$(repeat('=',39)) %s20 $(repeat('=',39))\n\n", title)
    @printf(
        "%12s  %5s  %3s  %2s  %23s  %3s  %8s  %10s  %7s  %s\n",
        "Test",
        "n",
        "id",
        "th",
        "algorithm",
        "it",
        "st",
        "time",
        "infeas",
        "rel dif"
    )
    println('-'^98)
end

function print_test(p, id, m, nthreads, iter, status, time, infeas, extra_info)
    @printf(
        "%12s  %5.0e  %3d  %2d  %23s  %3d  %8s  %10s  %7.1e  %7.1e\n",
        p.name,
        p.n,
        id,
        nthreads,
        m.name,
        iter,
        status,
        fnanosec(time),
        infeas,
        extra_info
    )
end

# Has the test already been run?
function executed(results, p, id, m, nthreads)
    if !isempty(
        results[
            (results.Instance .== p.name) .& (results.n .== p.n) .& (results.id .== id) .& (results.Algorithm .== m.name) .& (results.threads .== nthreads),
            :
        ]
    )
        @printf(
            "%12s  %5.0e  %3d  %2d  %23s already executed. Skipping...\n",
            p.name,
            p.n,
            id,
            nthreads,
            m.name
        )
        return true
    end
    return false
end

# Main function
function main(args)
    Random.seed!(0)
    nthreads = Threads.nthreads()

    # Get command line parameters
    opts = get_parameters()

    # Output DataFrame / file
    output = "results.jld2"
    results = DataFrame(
        [
            "Instance"=>String[];
            "n"=>Int64[];
            "id"=>Int64[];
            "Algorithm"=>String[];
            "threads"=>Int64[];
            "iter"=>Int64[];
            "st"=>[];
            "time"=>Float64[];
            "infeas"=>Float64[];
            "extra_info"=>Float64[]
        ]
    )

    if opts["continue"] && isfile(output)
        results = jldopen(output, "r")["results"]
    end

    for test in TESTS
        for p in test.instances
            print_test_header(test.title)

            for id in 1:opts["nreps"]
                # Create the problem
                origP = p.generate(p.n)

                # Run methods
                for m in test.methods
                    if executed(results, p, id, m, nthreads)
                        # Already done, skip...
                        continue
                    end

                    # Convert origP
                    P = m.convert(origP)

                    # Collect information
                    time = m.time(P, nthreads)

                    if !isinf(time)
                        infeas = m.infeas(P, nthreads)
                        extra_info = m.extra_info(P, nthreads)
                        iter, status = m.it_st(P, nthreads)

                        row = [
                            p.name;
                            p.n;
                            id;
                            m.name;
                            nthreads;
                            iter;
                            status;
                            time;
                            infeas;
                            extra_info
                        ]
                        push!(results, row)

                        print_test(
                            p, id, m, nthreads, iter, status, time, infeas, extra_info
                        )
                    end
                end

                # Save partial output binary file whenever all algorithms finish
                jldsave(output; results)
            end

            println(repeat('-', 95))
        end
    end

    return 0
end

# Run main if non-iteractive
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
