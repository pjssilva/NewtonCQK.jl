# Single-thread comparison of Condat's original code, the same method with the
# free variables tracked by their indices (simplexproj_Condat_idx), and
# NewtonCQK.jl, with dense and sparse outputs. Run it with one thread, e.g.
#
#   julia --project -t 1 condat_idx/runtests.jl
#
# Timing follows the other benchmarks: minimum over BenchmarkTools samples for
# each instance; condat_idx/results.jl takes the median over instances.
using NewtonCQK, BenchmarkTools, SparseArrays, Random, Printf, ArgParse

include("../third_party/condat/condat_interface.jl")

projectpath = isfile("Project.toml") ? "./" : "../"

function get_parameters()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nreps"
        arg_type = Int
        default = 20
        help = "number of random problems for each type and size"
        "--maxn"
        arg_type = Int
        default = 1_000_000_000
        help = "largest problem size"
    end
    return parse_args(s)
end

function estimatetime(b)
    b.params.gctrial = true
    b.params.gcsample = false
    b.params.evals = 1
    b.params.samples = 10000
    b.params.seconds = 2.0
    return minimum(run(b).times)   # ns
end

function rand_nonzerovec(gen, n)
    x = gen(n)
    while any(x .== 0.0)
        x .= gen(n)
    end
    return x
end

params = get_parameters()
NINST = params["nreps"]
MAXN = params["maxn"]
OUT = joinpath(projectpath, "results", "results_condat_idx.csv")

gens = [("Random 1", n -> rand(n)), ("Random 2", n -> randn(n)), ("Random 3", n -> 1e-3 .* randn(n))]
sizes = filter(<=(MAXN), [10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9])
methods = ["Condat", "Condat idx", "Newton dense", "Condat idx sparse", "Newton sparse"]

Random.seed!(1234)
open(OUT, "w") do io
    println(io, "instance,n,rep,method,time_ns,nnz")
    for (name, gen) in gens, n in sizes
        for rep in 1:NINST
            y = rand_nonzerovec(gen, n)
            sol = similar(y)
            work = Vector{Cuint}(undef, n)
            chunks = initialize_chunks(n; nchunks=1)
            nz = nnz(condat_idx_spproj(y, work))
            t = Dict{String,Float64}()
            t["Condat"] = estimatetime(@benchmarkable condat_proj!($sol, $y))
            t["Condat idx"] = estimatetime(@benchmarkable condat_idx_proj!($sol, $y))
            t["Newton dense"] = estimatetime(@benchmarkable simplex_proj!($sol, $y; chunks=$chunks))
            t["Condat idx sparse"] = estimatetime(@benchmarkable condat_idx_spproj($y, $work))
            t["Newton sparse"] = estimatetime(@benchmarkable spsimplex_proj($y; chunks=$chunks))
            for m in methods
                println(io, "$name,$n,$rep,$m,$(t[m]),$nz")
            end
            flush(io)
            @printf("%-8s n=%.0e rep=%d  nnz=%6d  %s\n", name, n, rep, nz,
                    join([@sprintf("%s=%.3f ms", m, t[m] / 1e6) for m in methods], "  "))
            y = sol = work = chunks = nothing
            GC.gc()
        end
    end
end
