using DataFrames
using JLD2
using Plots
using BenchmarkProfiles
using Printf
using Latexify
using Statistics
using Format
using LaTeXStrings

# read JLD2 file
function read_jld(filename)
    if isfile(filename)
        return jldopen(filename, "r")["results"]
    else
        error("JLD2 file not found")
    end
end

# construct a filtered table from data.
# algs is the vector of INDICES of algorithms to be considered
# testname is the STRING of the test to be considered
# n=0, threads=0, algs=[] or testname="" indicate that no corresponding filter is applied
# set latex=true to store formated numbers as strings (for LaTeX) and bold best values
function matrix(
    test_id; n=0, minthreads=1, maxthreads=100, algs=[], testname="", latex=true, bold=false
)
    # read results file
    allres = read_jld("output/results_$(TESTS[test_id].name).jld2")

    if isempty(algs)
        algs = 1:(size(allres, 2) - 4)
    end

    # filter results
    res = allres[
        (n > 0 ? allres.n .== n : allres.n .>= 0) .& (allres.threads .>= minthreads) .& (allres.threads .<= maxthreads) .& (if !isempty(
            testname
        )
            allres.Test .== testname
        else
            allres.Test .>= ""
        end),
        [1:4; algs .+ 4]
    ]

    # consolidates runs of the same test
    gres = groupby(res, [:Test, :n, :threads])
    cres = combine(gres, 5:size(res, 2) .=> Ref; renamecols=false)

    # construct the final matrix
    T = Matrix{Any}(undef, size(cres, 1) + 1, 3 + 2*(size(cres, 2) - 3))
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
function tex_table(test_id; n=0, minthreads=1, maxthreads=100, algs=[], testname="")
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
        testname=testname,
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
    maxthreads=100,
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

# plot of "number threads" x "time"
function plot_threads(
    test_id,
    n;
    testname="",
    algs=[],
    title="",
    alglabels=String[],
    testlabels=Dict(),
    legpos=:best
)
    @assert n > 0 "n must be > 0"
    if !isdir("output")
        mkdir("output")
    end
    output = "output/threads_$(TESTS[test_id].name)_$(n).pdf"
    T = matrix(test_id; n=n, algs=algs, testname=testname, latex=false)
    if isempty(algs)
        algs = 1:Int((size(T, 2) - 3) / 2)
    end
    if testname == ""
        testnames = unique(T[2:end, 1])
    else
        testnames = [testname]
    end
    if isempty(alglabels)
        alglabels = Vector{String}(undef, maximum(algs))
        for a in algs
            alglabels[a] = TESTS[test_id].algnames[a]
        end
    end
    if isempty(testlabels)
        for t in testnames
            push!(testlabels, t => t)
        end
    end
    fig = plot(; title=title, xlabel="number of threads", ylabel="time (ms)")
    for t in testnames
        T = matrix(test_id; n=n, testname=t, latex=false)
        # distinct number of threads in T
        threads = unique(T[2:end, 3])
        for a in algs
            if length(algs) > 1
                label = "$(alglabels[a]), $(testlabels[t])"
            else
                label = "$(testlabels[t])"
            end
            fig = plot!(
                1:length(threads),
                Float64.(T[2:end, 3 + 2 * a]);
                label=label,
                xticks=(1:length(threads), threads),
                legend=legpos,
                markershape=:diamond,
                markersize=4,
                lw=1
            )
        end
    end
    savefig(fig, output)
    println("File $(output) was generated. Rename it if necessary.")
end

# relative speedup
function plot_speedup(
    test_id,
    n,
    basealg;
    testname="",
    algs=[],
    title="",
    alglabels=String[],
    testlabels=Dict(),
    legpos=:best
)
    @assert n > 0 "n must be > 0"
    if !isdir("output")
        mkdir("output")
    end
    if testname == ""
        output = "output/speedup_$(TESTS[test_id].name)_$(n).pdf"
    else
        output = "output/speedup_$(TESTS[test_id].name)_$(n)_$(replace(testname, " " => "")).pdf"
    end
    T = matrix(test_id; n=n, algs=algs, testname=testname, latex=false)
    if isempty(algs)
        algs = 1:Int((size(T, 2) - 3) / 2)
    end
    if testname == ""
        testnames = unique(T[2:end, 1])
    else
        testnames = [testname]
    end
    if isempty(alglabels)
        alglabels = Vector{String}(undef, maximum(algs))
        for a in algs
            alglabels[a] = TESTS[test_id].algnames[a]
        end
    end
    if isempty(testlabels)
        for t in testnames
            push!(testlabels, t => t)
        end
    end
    fig = plot(; title=title, xlabel="number of threads", ylabel="relative speedup")
    for t in testnames
        T = matrix(test_id; n=n, testname=t, minthreads=2, latex=false)
        # distinct number of threads in T
        threads = sort(unique(T[2:end, 3]))
        basetime = matrix(
            test_id; n=n, algs=basealg, minthreads=1, maxthreads=1, testname=t, latex=false
        )[
            2, 5
        ]
        for a in algs
            if length(algs) > 1
                if length(testlabels) > 1
                    label = "$(alglabels[a]), $(testlabels[t])"
                else
                    label = "$(alglabels[a])"
                end
            else
                label = "$(testlabels[t])"
            end
            fig = plot!(
                1:length(threads),
                basetime ./ Float64.(T[2:end, 3 + 2 * a]);
                label=label,
                xticks=(1:length(threads), threads),
                legend=legpos,
                markershape=:diamond,
                markersize=4,
                lw=1
            )
        end
    end
    savefig(fig, output)
    println("File $(output) was generated. Rename it if necessary.")
end

# plot of "n" x "time"
function plot_n(
    test_id, testname; algs=[], title="", alglabels=String[], legpos=:best, minthreads=1
)
    @assert !isempty(testname) "testname must be given"
    if !isdir("output")
        mkdir("output")
    end
    output = "output/n_$(TESTS[test_id].name)_$(replace(testname, " " => "")).pdf"
    T = matrix(test_id; algs=algs, testname=testname, latex=false)
    if isempty(algs)
        algs = 1:Int((size(T, 2) - 3) / 2)
    end
    if isempty(alglabels)
        alglabels = Vector{String}(undef, maximum(algs))
        for a in algs
            alglabels[a] = TESTS[test_id].algnames[a]
        end
    end
    fig = plot(; title=title, xlabel=L"n", ylabel="time (ms)")
    ns = unique(T[2:end, 2])
    threads = unique(T[2:end, 3])
    for th in threads
        for a in algs
            if TESTS[test_id].cuda[a]
                if th > 1
                    # skip gpu when threads > 1
                    continue
                end
                label = "$(alglabels[a])"
            else
                if th < minthreads
                    continue
                end
                if th == 1
                    label = "$(alglabels[a]) (1 thread)"
                else
                    label = "$(alglabels[a]) ($(th) threads)"
                end
            end
            times = Float64[]
            for n in ns
                T = matrix(
                    test_id;
                    n=n,
                    minthreads=th,
                    maxthreads=th,
                    algs=a,
                    testname=testname,
                    latex=false
                )
                if size(T, 1) != 2
                    @error "$(a), $(th) threads, n=$(n) not found or returned more than one row"
                    return nothing
                end
                push!(times, T[2, 5])
            end
            fig = plot!(
                1:length(ns),
                times;
                label=label,
                xticks=(1:length(ns), format.(ns; commas=true)),
                legend=legpos,
                markershape=:diamond,
                markersize=4,
                lw=1,
                yaxis=:log
            )
        end
    end
    savefig(fig, output)
    println("File $(output) was generated. Rename it if necessary.")
end

# function table_cpu_gpu(test_id, testname; algs=[])
#     if !isdir("output")
#         mkdir("output")
#     end
#     output = "output/table_$(TESTS[test_id].name)_$(replace(testname, " " => "")).tex"
#     T = matrix(test_id; testname=testname, algs=algs)
#     ns = sort(unique(T[2:end,2]))
#     ths = sort(unique(T[2:end,3]))
#
#     outT =
#     for
#
#     tex = open(output, "w")
#     write(tex, latexify(T, env=:table, latex=false))
#     close(tex)
#     println("File $(output) was generated. Rename and adjust it if necessary.")
# end

function generate_all()
    ###################
    # cqk: speedup
    ###################
    plot_speedup(
        1,
        1000,
        1;
        alglabels=["Algorithm 1"],
        testlabels=Dict(
            "uncorr"=>"Uncorrelated",
            "weakly corr"=>"Weakly Correlated",
            "corr"=>"Correlated"
        ),
        title=L"n = 1,000"
    )

    plot_speedup(
        1,
        10000,
        1;
        alglabels=["Algorithm 1"],
        testlabels=Dict(
            "uncorr"=>"Uncorrelated",
            "weakly corr"=>"Weakly Correlated",
            "corr"=>"Correlated"
        ),
        title=L"n = 10,000",
        legpos=:bottomleft
    )

    plot_speedup(
        1,
        100000,
        1;
        alglabels=["Algorithm 1"],
        testlabels=Dict(
            "uncorr"=>"Uncorrelated",
            "weakly corr"=>"Weakly Correlated",
            "corr"=>"Correlated"
        ),
        title=L"n = 100,000"
    )

    plot_speedup(
        1,
        1000000,
        1;
        alglabels=["Algorithm 1"],
        testlabels=Dict(
            "uncorr"=>"Uncorrelated",
            "weakly corr"=>"Weakly Correlated",
            "corr"=>"Correlated"
        ),
        title=L"n = 1,000,000"
    )

    ###################
    # cqk: n x time, comparison between cpu and gpu
    ###################
    plot_n(2, "uncorr"; alglabels=["CPU", "GPU"])#, title="Uncorrelated")
    #plot_n(2, "weakly corr", alglabels=["CPU", "GPU"], title="Weakly Correlated")
    #plot_n(2, "corr", alglabels=["CPU", "GPU"], title="Correlated")

    ###################
    # simplex: performance profile for sequential algorithms
    ###################
    pp(
        3;
        minthreads=1,
        maxthreads=1,
        title="",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "Algorithm 3 (Condat's implementation)",
            "Dai and Chen's algorithm"
        ]
    )

    ###################
    # simplex: speedup
    # [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    ###################
    plot_speedup(
        3,
        100_000,
        2;
        algs=[1; 2; 4],
        testname="Random 1",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:topleft,
        title=L"n = 10^5, \textnormal{Proj1}"
    )

    plot_speedup(
        3,
        1_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 1",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:topleft,
        title=L"n = 10^6, \textnormal{Proj1}"
    )

    plot_speedup(
        3,
        10_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 1",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:topleft,
        title=L"n = 10^7, \textnormal{Proj1}"
    )

    plot_speedup(
        3,
        100_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 1",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:bottomright,
        title=L"n = 10^8, \textnormal{Proj1}"
    )

    plot_speedup(
        3,
        1_000_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 1",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:bottomright,
        title=L"n = 10^9, \textnormal{Proj1}"
    )

    plot_speedup(
        3,
        100_000,
        2;
        algs=[1; 2; 4],
        testname="Random 2",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:topleft,
        title=L"n = 10^5, \textnormal{Proj2}"
    )

    plot_speedup(
        3,
        1_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 2",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:topleft,
        title=L"n = 10^6, \textnormal{Proj2}"
    )

    plot_speedup(
        3,
        10_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 2",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:topleft,
        title=L"n = 10^7, \textnormal{Proj2}"
    )

    plot_speedup(
        3,
        100_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 2",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:bottomright,
        title=L"n = 10^8, \textnormal{Proj2}"
    )

    plot_speedup(
        3,
        1_000_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 2",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:bottomright,
        title=L"n = 10^9, \textnormal{Proj2}"
    )

    plot_speedup(
        3,
        100_000,
        2;
        algs=[1; 2; 4],
        testname="Random 3",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:topleft,
        title=L"n = 10^5, \textnormal{Proj3}"
    )

    plot_speedup(
        3,
        1_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 3",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:topleft,
        title=L"n = 10^6, \textnormal{Proj3}"
    )

    plot_speedup(
        3,
        10_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 3",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:topleft,
        title=L"n = 10^7, \textnormal{Proj3}"
    )

    plot_speedup(
        3,
        100_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 3",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:bottomright,
        title=L"n = 10^8, \textnormal{Proj3}"
    )

    plot_speedup(
        3,
        1_000_000_000,
        2;
        algs=[1; 2; 4],
        testname="Random 3",
        alglabels=[
            "Algorithm 4 (dense solution)",
            "Algorithm 4 (sparse solution)",
            "",
            "Dai and Chen's algorithm"
        ],
        legpos=:bottomright,
        title=L"n = 10^9, \textnormal{Proj3}"
    )
end

show_tests()
