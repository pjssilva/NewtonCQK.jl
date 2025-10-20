using DataFrames
using JLD2
using Plots
using BenchmarkProfiles
using Printf
using Latexify
using Statistics
using Format
using LaTeXStrings
using OpenML

# Instances names and sizes from tests_cqk.jl and tests_simplex.jl
# We do not include these files here because to avoid redefine TESTS structure.
# Also, we are freedom to select a subset of tests here.
CQK_names = ["uncorr", "weakly corr", "corr"]
CQK_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

simplex_names = ["Random 1", "Random 2", "Random 3"]
simplex_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

# Algorithm legend labels
alglabels = Dict(
    "cqk (CPU, FP64)"       => "Algorithm 1",
    "cqn"                   => "CMS",
    "cqk (GPU, FP64)"       => "Algorithm 1 (GPU)",
    "cqk (CPU, FP32)"       => "Algorithm 1 (dense solution)",
    "cqk (GPU, FP32)"       => "Algorithm 1 (GPU)",
    "simplex (CPU, FP64)"   => "Algorithm 4 (dense solution)",
    "sp simplex (CPU, FP64)"=> "Algorithm 4 (sparse solution)",
    "Condat C"              => "Condat (Algorithm 3)",
    "P Condat (simplex)"    => "Dai and Chen's algorithm",
    "simplex (GPU, FP64)"   => "Algorithm 4 (GPU)",
    "simplex (CPU, FP32)"   => "Algorithm 4 (dense solution)",
    "simplex (GPU, FP32)"   => "Algorithm 4 (GPU)",
    "l1ball (CPU, FP64)"    => "Specialized Algorithm 1 (dense)",
    "sp l1ball (CPU, FP64)" => "Specialized Algorithm 1 (sparse)",
    "P Condat (l1ball)"     => "Dai and Chen's algorithm",
    "l1ball (GPU, FP64)"    => "Specialized Algorithm 1 (GPU)",
    "l1ball (CPU, FP32)"    => "Specialized Algorithm 1 (dense)",
    "l1ball (GPU, FP32)"    => "Specialized Algorithm 1 (GPU)"
)

# Instance legend labels
instancelabels = Dict(
    "uncorr"        => "Uncorrelated",
    "weakly corr"   => "Weakly correlated",
    "corr"          => "Correlated"
)

# read results in JLD2 file
function read_results(filenames)
    if isa(filenames, String)
        files = [filenames]
    else
        files = filenames
    end

    res = []

    for f in files
        @assert isfile(f) "$(f) file not found"

        # read JLD2 file
        jld2file = jldopen(f, "r")
        allres = read(jld2file, "results")
        close(jld2file)

        # consolidates runs
        gres = groupby(allres, [:Instance, :n, :threads, :Algorithm])

        # mean of iter
        # median of runtimes
        # mean of infeasibilities
        # solved problems count
        if isempty(res)
            res = combine(gres,
                    [:iter; :time; :infeas; :st] .=> [mean; median; mean; x -> count(x .== :solved)];
                    renamecols=false
                    )
        else
            res = vcat(res, combine(gres,
                    [:iter; :time; :infeas; :st] .=> [mean; median; mean; x -> count(x .== :solved)];
                    renamecols=false
                    ))
        end
    end

    return res
end

function filter_results(allres;
    instance="", n=0, minthreads=1, maxthreads=500, algorithm=""
)
    return allres[
        (!isempty(instance) ? allres.Instance .== instance : allres.Instance .>= "") .&
        (n > 0 ? allres.n .== n : allres.n .>= 0) .&
        (allres.threads .>= minthreads) .&
        (allres.threads .<= maxthreads) .&
        (!isempty(algorithm) ? allres.Algorithm .== algorithm : allres.Algorithm .>= "") .&
        (allres.st .> 0)
        , :
    ]
end

# construct a filtered table from data
# algs is the vector of INDICES of algorithms to be considered
# n=0, threads=0 or, algs=[] indicate that no corresponding filter is applied
# set latex=true to store formated numbers as strings (for LaTeX) and bold best values
function matrix(
    test_id; n=0, minthreads=1, maxthreads=500, algs=[], latex=true, bold=false
)
    # read results
    allres = read_results("results.jld2")

    if isempty(algs)
        algs = unique(res.Algorithm)
    end

    # filter results
    res = allres[
        (n > 0 ? allres.n .== n : allres.n .>= 0) .&
        (allres.threads .>= minthreads) .&
        (allres.threads .<= maxthreads)
        , :
    ]

    # construct the final matrix
    T = Matrix{Any}(undef, size(res, 1) + 1, 3 + 2*(size(res, 2) - 3))
    T[1, 1:3] = ["Test"; "n"; "threads"]
    c = 4
    for a in algs
        T[1, c] = "$(TESTS[test_id].algnames[a]) iter"
        T[1, c + 1] = "$(TESTS[test_id].algnames[a]) time"
        c += 2
    end
    fmt_n = generate_formatter("%'d")
    fmt_iter = generate_formatter("%6.2lf")
    fmt_time = generate_formatter("%8.2e")
    for l in 1:size(cres, 1)
        row = Vector(cres[l, 1:3])
        if latex
            row[2] = fmt_n(row[2])
        end
        best = Inf
        ibest = 0
        for c in 4:size(cres, 2)
            means = mean(reinterpret(reshape, Float64, cres[l, c]); dims=2)[:]
            # transform to miliseconds (ms)
            means[2] *= 1e-6
            if latex
                append!(row, [fmt_iter(means[1]); fmt_time(means[2])])
            else
                append!(row, means)
            end
            if means[2] < best
                best = means[2]
                ibest = length(row)
            end
        end
        if latex && bold
            # bold best value
            row[ibest] = "\\bf $(row[ibest])"
        end
        T[l + 1, :] .= row
    end

    return T
end

# write LaTeX file of a table
function tex_table(test_id; n=0, minthreads=1, maxthreads=500, algs=[], testname="")
    if !isdir("output")
        mkdir("output")
    end
    if testname == ""
        output = "output/table_$(TESTS[test_id].name).tex"
    else
        output = "output/table_$(TESTS[test_id].name)_$(replace(testname, " " => "")).tex"
    end
    T = matrix(
        test_id;
        n=n,
        minthreads=minthreads,
        maxthreads=maxthreads,
        algs=algs
    )
    ordT = sortslices(T; dims=1, by=row -> (row[1], row[2]))
    tex = open(output, "w")
    texcode = latexify(ordT; env=:table, latex=false)
    texcode = replace(texcode, "e+" => "e\$+\$")
    texcode = replace(texcode, "e-" => "e\$-\$")
    write(tex, texcode)
    close(tex)
    println("File $(output) was generated. Rename and adjust it if necessary.")
end

# plot the figure of a performance profile of CPU times
# IMPORTANT: unless time = Inf, it is considered that all algorithms solved all problems.
function pp(
    test_id;
    n=0,
    minthreads=1,
    maxthreads=500,
    algs=[],
    title="CPU time (ms)",
    alglabels=String[]
)
    if !isdir("output")
        mkdir("output")
    end
    output = "output/pp_$(TESTS[test_id].name).pdf"
    T = matrix(test_id; n=n, minthreads=minthreads, maxthreads=maxthreads, latex=false)
    if isempty(algs)
        algs = 1:Int((size(T, 2) - 3) / 2)
    end
    if isempty(alglabels)
        alglabels = Vector{String}(undef, maximum(algs))
        for a in algs
            alglabels[a] = TESTS[test_id].algnames[a]
        end
    end
    # capture the names of algorithms from TESTS
    algnames = String[]
    Tcols = Int64[]
    for a in algs
        push!(algnames, TESTS[test_id].algnames[a])
        push!(Tcols, 3 + 2*a)
    end
    fig = performance_profile(
        PlotsBackend(),
        Float64.(T[2:end, Tcols]),
        alglabels;
        title=title,
        logscale=true,
        #fontfamily="Computer Modern",
        lw=2#, color=:black, ls=:auto
    )
    savefig(fig, output)
    println("File $(output) was generated. Rename it if necessary.")
end

# relative speedup
# if 'inst' is a vector, plot speedups relative the algorithm 'alg'
# if 'alg' is a vector, plot speedups relative the instance 'inst'
function plot_speedup(
    inst,
    n,
    basealg,
    alg;
    title="",
    legpos=:best,
    minthreads=2,
    plot_basealg=true,
    output="",          # additional identifier for output files
    algcuda="",
    filenames="results.jld2"
)
    @assert !isempty(inst) "inst must be non empty"
    @assert n > 0 "n must be > 0"
    @assert !isempty(basealg) "basealg must be non empty"
    @assert !isempty(alg) "alg must be non empty"
    @assert (length(alg) == 1) || (length(inst) == 1) "alg or inst must be length = 1"
    @assert (length(alg) > 1) || (length(inst) > 1) "alg or inst must be length > 1"

    if !isdir("output")
        mkdir("output")
    end
    if length(alg) > 1
        outfile = "output/speedup_$(output)_$(inst[1])_$(n).pdf"
    else
        outfile = "output/speedup_$(output)_$(alg[1])_$(n).pdf"
    end
    res = read_results(filenames)

    # initialize plot
    fig = plot(; title=title,
               xlabel="number of threads",
               ylabel="relative speedup",
               legend=legpos,
               fontfamily="Computer Modern"
          )

    maxth = 1

    # plot speedup graph for each algorithm or instance
    for k in 1:max(length(alg), length(inst))
        # read particular results
        if length(alg) > 1
            ii = inst[1]
            aa = alg[k]
        else
            ii = inst[k]
            aa = alg[1]
        end
        T = filter_results(res; instance=ii, n=n, minthreads=minthreads, algorithm=aa)
        if isempty(T)
            # none instance were solved!
            return
        end
        # base runtime
        bt = filter_results(res; instance=ii, n=n,
                            maxthreads=1, algorithm=basealg)
        if isempty(bt)
            # test not found for the base algorithm
            return
        end
        basetime = bt[1,:time]

        # distinct number of threads
        threads = Int64.(sort(unique(T.threads)))
        maxth = max(maxth, length(threads))

        # add plot
        fig = plot!(
            1:length(threads),
            basetime ./ Float64.(T[:, :time]);
            label=(length(alg) > 1) ? alglabels[alg[k]] : instancelabels[inst[k]],
            xticks=(1:length(threads), threads),
            markershape=:diamond,
            markersize=4,
            lw=1
        )
    end

    # cuda (only takes effect if plot is relative to algorithms)
    if !isempty(algcuda) && (length(inst) == 1)
        cudat = filter_results(res; instance=inst[1], n=n,
                            maxthreads=1, algorithm=algcuda)
        if !isempty(cudat)
            cudatime = cudat[1,:time]
            fig = plot!(
                1:maxth,
                fill(cudatime, maxth);
                label=alglabels[algcuda],
                markershape=:none,
                ls=:dash,
                lw=1
            )
        end
    end

    # base algorithm
    if plot_basealg
        fig = plot!(
            1:maxth,
            fill(1.0, maxth);
            label=alglabels[basealg],
            markershape=:none,
            ls=:dot,
            lw=1
        )
    end
    savefig(fig, outfile)
    println("File $(outfile) was generated.")
end

function generate_all()
    ###################
    # Speedup CQK
    ###################
    base = "cqk (CPU, FP64)"
    for n in CQK_sizes
        plot_speedup(
            CQK_names,
            n,
            base,
            ["cqk (CPU, FP64)"];
            title=latexstring("n = 10^{$(ceil(Int64, log10(n)))}"),
            plot_basealg=false,
            filenames=["results.jld2", "results_cpu.jld2"]
        )
    end

    ###################
    # Speedup Simplex
    ###################
    base = "Condat C"
    for n in simplex_sizes, p in simplex_names
        ptext = replace(p," " => "\\ ")
        plot_speedup(
            [p],
            n,
            base,
            ["simplex (CPU, FP64)", "sp simplex (CPU, FP64)", "P Condat (simplex)"];
            title=latexstring("n = 10^{$(ceil(Int64, log10(n)))}, \\textnormal{$(ptext)}"),
            # include 1 thread, as the comparison is with Condat's C code
            minthreads=1,
            #algcuda="simplex (GPU, FP64)",
            filenames=["results.jld2", "results_cpu.jld2"]
        )
    end
end

# LaTex table datasets
function table_datasets()
    jld2file = jldopen("svm_param.jld2", "r")
    param = read(jld2file, "param")
    close(jld2file)

    datasets = OpenML.list_datasets(
        tag = "uci",
        output_format = DataFrame
    )

    if !isdir("output")
        mkdir("output")
    end
    output = "output/table_datasets.tex"
    fmt = generate_formatter("%6.2lf")
    fmt_n = generate_formatter("%'d")
    tex = open(output, "w")
    texcode = ""
    for k in keys(param)
        ni = datasets[datasets.name .== k, :NumberOfInstances][1]
        nf = datasets[datasets.name .== k, :NumberOfFeatures][1] - 1
        gamma, C = param[k]
        texcode = texcode * "\\texttt{$(k)} & $(fmt_n(ni)) & $(fmt_n(nf)) & $(fmt(gamma)) & $(fmt(C)) \\\\"
    end
    texcode = replace(texcode, "e+" => "e\$+\$")
    texcode = replace(texcode, "e-" => "e\$-\$")
    write(tex, texcode)
    close(tex)
    println("File $(output) was generated.")
end

# Run main if non-iteractive
if abspath(PROGRAM_FILE) == @__FILE__
    generate_all()
end
