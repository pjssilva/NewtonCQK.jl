using DataFrames
using JLD2
using Plots
using Printf
using Latexify
using Statistics
using Format
using LaTeXStrings

include("../common/jld2_read.jl")
include("../svm/dataset.jl")

projectpath = isfile("Project.toml") ? "./" : "../"
output_path = joinpath(projectpath, "results", "output")

if !isdir(output_path)
    mkdir(output_path)
end

# Formats
fmt_d = generate_formatter("%'d")
fmt_lf = generate_formatter("%6.2lf")
fmt_lf1 = generate_formatter("%5.1lf")
fmt_e = generate_formatter("%8.2e")
fmt_e0 = generate_formatter("%7.0e")
fmt_etex(v) = replace(fmt_e(v), "e+" => "e\$+\$")
fmt_etex0(v) = replace(fmt_e0(v), "e+" => "e\$+\$")

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
    "l1ball (GPU, FP32)"    => "Specialized Algorithm 1 (GPU)",
    "l1ball (bp)"           => "Our algorithm",
    "l1ball (bp) x0"        => "Our algorithm (warm start)",
    "cqk (SVM)"             => "Algorithm 1",
    "cqk (SVM) x0"          => "Algorithm 1 (warm start)",
    "cqn (SVM)"             => "CMS algorithm"
)

#########################
# RANDOM TESTS
#########################

# Instances names and sizes from tests_cqk.jl and tests_simplex.jl
# We do not include these files here because to avoid redefine TESTS structure.
# Also, we are freedom to select a subset of tests here.
CQK_names = ["uncorr", "weakly corr", "corr"]
CQK_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

simplex_names = ["Random 1", "Random 2", "Random 3"]
simplex_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]

# Instance legend labels
instancelabels = Dict(
    "uncorr"        => "Uncorrelated",
    "weakly corr"   => "Weakly correlated",
    "corr"          => "Correlated"
)

# read results in the JLD2 file
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
        allres = jld2_read("results", f)

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
    instance="", minn=0, maxn=10^20, minthreads=1, maxthreads=500, algorithm=""
)
    return allres[
        (!isempty(instance) ? allres.Instance .== instance : allres.Instance .>= "") .&
        (allres.n .>= minn) .&
        (allres.n .<= maxn) .&
        (allres.threads .>= minthreads) .&
        (allres.threads .<= maxthreads) .&
        (!isempty(algorithm) ? allres.Algorithm .== algorithm : allres.Algorithm .>= "") .&
        (allres.st .> 0)
        , :
    ]
end

# write LaTeX file of a table
function table_cpu_gpu(
    inst,
    cpualg,
    gpualg;
    minn = 0,
    maxn = Inf,
    maxthreads = 500,
    suffix="",
    filenames=joinpath(projectpath, "results", "results_random.jld2")
)
    @assert length(cpualg) == length(gpualg) "Lists of CPU and GPU algorithms must have the same size"

    outfile = joinpath(output_path, "table$(suffix).tex")

    res = read_results(filenames)
    res_tmp = filter_results(res; instance=inst[1])
    ns = sort(Int64.(unique(res_tmp[:,:n])))
    threads = sort(Int64.(unique(res_tmp[:,:threads])))

    tex = open(outfile, "w")
    write(tex, "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep\\fill}ll$(repeat("ll", length(inst)))@{\\extracolsep\\fill}}\n")
    write(tex, "\\hline\n")
    write(tex, "& ")
    for p in inst
        write(tex, " & \\multicolumn{2}{l}{$(instancelabels[p])}")
    end
    write(tex, "\\\\\n")
    write(tex, "\$n\$ & th $(repeat(" & CPU time & GPU sp up", length(inst)))\\\\\n")
    write(tex, "\\hline\n")
    for n in ns
        if (n < minn) || (n > maxn)
            continue
        end
        for th in threads
            if th > maxthreads
                continue
            end
            write(tex, "\$10^{$(ceil(Int64, log10(n)))}\$ & $(th)")

            for p in inst
                Tgpu = filter_results(
                    res; minn=n, maxn=n, algorithm=gpualg, instance=p
                )
                Tcpu = filter_results(
                    res; minn=n, maxn=n, minthreads=th, maxthreads=th,
                    algorithm=cpualg, instance=p
                )
                if isempty(Tcpu)
                    write(tex, " & -- & $(!isempty(Tgpu) ? "\\checkmark" : "--") ")
                else
                    cputime = Tcpu[1,:time][1]
                    cpuiter = Tcpu[1,:iter][1]
                    write(tex, " & $(fmt_etex0(cputime)) ($(strip(fmt_lf1(cpuiter))))")
                    if !isempty(Tgpu)
                        gputime = Tgpu[1,:time][1]
                        gpuiter = Tgpu[1,:iter][1]
                        spup = cputime / gputime
                        write(tex, " & $(fmt_lf1(spup)) ($(strip(fmt_lf1(gpuiter))))")
                    else
                        write(tex, " & --")
                    end
                end
            end

            write(tex, " \\\\\n")
        end
        if n < min(ns[end], maxn)
            write(tex, "\\hline\n")
        end
    end
    write(tex, "\\hline\n\\end{tabular*}")

    close(tex)
    println("File $(outfile) was generated.")
end

# relative speedup
# if 'inst' is a vector, plot speedups relative to the algorithm 'alg'
# if 'alg' is a vector, plot speedups relative to the instance 'inst'
function plot_speedup(
    inst,
    n,
    basealg,
    alg;
    title="",
    legpos=:best,
    minthreads=2,
    plot_basealg=true,
    algcuda="",
    filenames=joinpath(projectpath, "results", "results_random.jld2")
)
    @assert !isempty(inst) "inst must be non empty"
    @assert n > 0 "n must be > 0"
    @assert !isempty(basealg) "basealg must be non empty"
    @assert !isempty(alg) "alg must be non empty"
    @assert (length(alg) == 1) || (length(inst) == 1) "alg or inst must be length = 1"
    @assert (length(alg) > 1) || (length(inst) > 1) "alg or inst must be length > 1"

    if length(alg) > 1
        outfile = joinpath(output_path, "speedup_$(replace(inst[1], " " => "_"))_$(n).pdf")
    else
        outfile = joinpath(output_path, "speedup_$(replace(alg[1], " " => "_"))_$(n).pdf")
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
        T = filter_results(
            res; instance=ii, minn=n, maxn=n,
            minthreads=minthreads, algorithm=aa
        )
        if isempty(T)
            # no instance solved!
            return
        end
        # base runtime
        bt = filter_results(
            res; instance=ii, minn=n, maxn=n,
            maxthreads=1, algorithm=basealg
        )
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
        cudat = filter_results(
            res; instance=inst[1], minn=n, maxn=n,
            maxthreads=1, algorithm=algcuda
        )
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


#########################
# BASIS PURSUIT AND SVM
#########################

function spg_plots(
    basealg,
    algs,
    instance;
    measure=:time,
    legpos=:best,
    prefix="",
    miniter=1,
    title="",
    threads=[1],
    rangesize=100,
    blanksize=20,
    xstep=0,
    filename=joinpath(projectpath, "results", "results_basis_pursuit.jld2")
)
    allres = jld2_read("results", filename)
    if isnothing(allres)
        return
    end

    allres = allres[allres.Instance .== instance,:]

    # the base is the algorithm with "threads[1]" threads
    res = allres[allres.threads .== threads[1],:]

    if isempty(res)
        return
    end

    if length(threads) == 1
        outfile = joinpath(output_path, "$(prefix)$(instance)_$(measure)_th$(threads[1]).pdf")
    else
        outfile = joinpath(output_path, "$(prefix)$(instance)_$(measure).pdf")
    end

    relative = (measure == :time)
    nfixed = (measure == :nfixed)

    # iters for plot
    iters = Int64.(unique(res.outiter))

    mask = union(
        1:min(rangesize,length(iters)),
        max(1,length(iters)-rangesize+1):length(iters)
    )

    miniter = min(miniter, length(iters))
    iters = iters[mask]

    # base algorithm
    basemeasures = Float64.(res[res.Algorithm .== basealg, measure])[mask]

    if xstep <= 0
        xstep = max(1, ceil(Int64, length(iters)/15))
    end

    if iters[end] > length(iters)
        # iters are not consecutive, we insert a blank space between the two intervals
        iterbreak = 1
        while iters[iterbreak] == iterbreak
            iterbreak += 1
        end
        # iters to be plotted
        plotiters = union(
            miniter:(iterbreak-1),
            (iterbreak+blanksize):length(iters).+blanksize
        )
        # do not plot the graph in the blank space
        novalues = [iterbreak]
        # positions of xsticks
        xticks = union(
            miniter:xstep:(iterbreak-1),
            (iterbreak+blanksize):xstep:length(iters).+blanksize
        )
        # texts of xsticks (may not conincide with positions)
        xtickstext = union(
            miniter:xstep:(iterbreak-1),
            iters[end-rangesize+1]:xstep:iters[end]
        )
    else
        plotiters = miniter:length(iters)
        novalues = Int64[]
        xticks = miniter:xstep:length(iters)
        xtickstext = xticks
    end

    # initialize plot
    fig = plot(; title=title,
        xlabel="SPG iteration",
        ylabel=nfixed ? "Fixed variables (%)" : relative ? "relative speedup" : "",
        legend=nfixed ? false : legpos,
        fontfamily="Computer Modern",
        xticks=(xticks, xtickstext)
    )

    basemeasures[novalues] .= NaN

    baseplot = relative ? fill(1.0, length(basemeasures)) : basemeasures
    baseplot[novalues] .= NaN

    if nfixed
        n = res[res.Algorithm .== basealg, :n][1]
        baseplot ./= n
    end

    # base alg
    fig = plot!(
        plotiters,
        baseplot[miniter:end],
        label = "$(length(threads) > 1 ? "1 thread" : "$(alglabels[basealg])")",
        markershape=:none,
        lw=1
    )

    if !nfixed
        if length(threads) == 1
            # specific number of threads, different algorithms
            for a in algs
                measures = Float64.(res[res.Algorithm .== a, measure])[mask]
                if minimum(measures) < 0
                    continue
                end
                measures[novalues] .= NaN

                fig = plot!(
                    plotiters,
                    relative
                        ?
                        basemeasures[miniter:end]./measures[miniter:end]
                        :
                        measures[miniter:end];
                    label=alglabels[a],
                    markershape=:none,
                    lw=1
                )
            end
        else
            # several threads, only base algorithm
            for t in threads[2:end]
                res = allres[(allres.Algorithm .== basealg) .& (allres.threads .== t),:]
                if isempty(res)
                    continue
                end

                # other algorithms following the order in "algs"
                measures = Float64.(res[:, measure])[mask]
                if minimum(measures) < 0
                    continue
                end
                measures[novalues] .= NaN

                fig = plot!(
                    plotiters,
                    relative
                        ?
                        basemeasures[miniter:end]./measures[miniter:end]
                        :
                        measures[miniter:end];
                    label="$(t) threads",
                    markershape=:none,
                    lw=1
                )
            end
        end
    end

    savefig(fig, outfile)
    println("File $(outfile) was generated.")
end


#########################
# GENERATE OUTPUTS
#########################

function generate_all()

    println("===================\nResults\n===================")

    ###################
    # Random, Speedup CQK
    ###################
    base = "cqk (CPU, FP64)"
    for n in CQK_sizes
        plot_speedup(
            CQK_names,
            n,
            base,
            ["cqk (CPU, FP64)"];
            title=latexstring("n = 10^{$(ceil(Int64, log10(n)))}"),
            plot_basealg=false
        )
    end

    ###################
    # Random, Speedup Simplex
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
            #algcuda="simplex (GPU, FP64)"
        )
    end

    ###################
    # Random, Tables CPU vs GPU
    ###################
    table_cpu_gpu(
        ["uncorr";"weakly corr";"corr"],
        "cqk (CPU, FP32)",      # CPU algorithm
        "cqk (GPU, FP32)",      # GPU algorithm
        suffix="_FP32",
        minn = 10^4,
        maxn = 10^9
    )
    table_cpu_gpu(
        ["uncorr";"weakly corr";"corr"],
        "cqk (CPU, FP64)",      # CPU algorithm
        "cqk (GPU, FP64)",      # GPU algorithm
        suffix="_FP64",
        minn = 10^4,
        maxn = 10^9
    )

    ###################
    # Basis pursuit
    ###################
    for n in [1;11]
        # Speedup, iterations and #fixed vars per thread
        for t in [1;2;4;8;16;24;48]
            for m in [:time; :iter; :nfixed]
                if m == :nfixed
                    title = "SC$(n)"
                elseif m == :time
                    title = "SC$(n), $(t) thread$(t > 1 ? "s" : "")"
                elseif m == :iter
                    title = "SC$(n), projection iterations ($(t) thread$(t > 1 ? "s" : ""))"
                end
                spg_plots(
                    "l1ball (bp)",
                    ["l1ball (bp) x0"; "P Condat (l1ball)"],
                    "SClog$(n).mat",
                    miniter=3,
                    title=title,
                    threads=[t],
                    measure=m,
                    filename=joinpath(projectpath, "results", "results_basis_pursuit.jld2"),
                    prefix="bp_",
                    #legpos=:topright
                )
            end
        end

        # Speedup, various threads
        spg_plots(
            "l1ball (bp) x0",
            ["l1ball (bp) x0"],
            "SClog$(n).mat",
            miniter=3,
            title="Our algorithm on SC$(n)",
            threads=[1;2;4;8;16;24;48],
            measure=:time,
            filename=joinpath(projectpath, "results", "results_basis_pursuit.jld2"),
            prefix="bp_"
        )
    end

    ###################
    # SVM
    ###################
    for p in ["mnist_784"; "cdc_diabetes"]
        problem = (p == "mnist_784") ? "MNIST" : "cdc_diabetes"

        # Speedup, iterations and #fixed vars per thread
        for t in [1;2;4;8;16;24;48]
            for m in [:time; :iter; :nfixed]
                if m == :nfixed
                    title = problem
                elseif m == :time
                    title = "$(problem), $(t) thread$(t > 1 ? "s" : "")"
                elseif m == :iter
                    title = "$(problem), projection iterations ($(t) thread$(t > 1 ? "s" : ""))"
                end
                spg_plots(
                    "cqk (SVM)",
                    (t == 1) ? ["cqk (SVM) x0"; "cqn (SVM)"] : ["cqk (SVM) x0"],
                    p,
                    miniter=3,
                    title=title,
                    threads=[t],
                    measure=m,
                    filename=joinpath(projectpath, "results", "results_svm.jld2"),
                    prefix="svm_",
                    #legpos=:topright
                )
            end
        end

        # Speedup, various threads
        spg_plots(
            "cqk (SVM) x0",
            ["cqk (SVM) x0"],
            p,
            miniter=3,
            title="Algorithm 1 on $(problem)",
            threads=[1;2;4;8;16;24;48],
            measure=:time,
            filename=joinpath(projectpath, "results", "results_svm.jld2"),
            prefix="svm_"
        )
    end
end

# Run main if non-iteractive
if abspath(PROGRAM_FILE) == @__FILE__
    generate_all()
end
