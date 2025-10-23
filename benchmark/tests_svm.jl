include("spg.jl")

# Filter datasets for binary classification
datasets = OpenML.list_datasets(
    #tag = "uci",
    filter="number_classes/2/number_instances/1000..100000/number_missing_values/0",
    output_format = DataFrame
)

function jld2_read(objectname, filename; test = true)
    output = nothing
    if isfile(filename) && test
        jld2file = jldopen(filename, "r")
        output = read(jld2file, objectname)
        close(jld2file)
    end
    return output
end

# Objective function
function f(x, H, Z, y)
    return 0.5 * (x' * H(Z) * x) - sum(sign.(y) .* x)
end

# Gradient of objective function
function g!(g, x, H, Z, y)
    g .= (H(Z) * x) .- sign.(y)
end

# Projection (solve particular CQK)
function proj!(p, z, x0, chunks, P::CQKProblem)
    P.a .= z
    _, flag = cqk!(p, P, chunks=(chunks), x0=(x0))
    return (flag == :solved)
end

# Apply SPG and optionally benchmark
# Training data is considered scaled here!
function svm_solve(
    instance, Z, y, nthreads;
    results = nothing, γ = 0.01, C = 1.0, verbose = 1
)
    @assert γ > 0.0 throw(ArgumentError("γ must be positive"))
    @assert C > 0.0 throw(ArgumentError("C must be positive"))
    @assert !isempty(instance) throw(ArgumentError("instance must be provided"))
    @assert size(Z,2) == length(y) throw(DimensionMismatch("Z and y have incompatible dimensions"))

    n = size(Z,2)

    # CQK for subproblems
    # Note: P.b must be positive, so we change variables (P.l, P.u, P.a must be
    # adjusted)
    P = CQKProblem(ones(n), zeros(n), ones(n), 0.0, zeros(n), fill(C,n))
    maskchg = (y .< 0.0)
    P.l[maskchg] .= -C
    P.u[maskchg] .= 0.0

    # The problem is, after changing variables,
    # min  0.5*x'Hx - s'x
    # s.t. |y'|x = 0
    #      0 <= xi <= C,  i: yi > 0
    #     -C <= xi <= 0,  i: yi < 0
    # where si = 1 if wi > 0 and si = -1 of yi < 0

    # The subproblem at iteration k is, after changing variables,
    # the problem above with H = I and s = xk - lambda*grad f

    # Kernel
    K(a,b) = exp(-γ * sqeuclidean(a,b))

    # "Lazy" Hessian
    H(Z) = @inbounds @views [ K(Z[:,i],Z[:,j]) for i=1:n, j=1:n ]

    tmp = similar(P.a)

    # Callback function for benchmarking
    # P is relative to x, which must be updated before by proj!. For benchmarking,
    # we start at x0 = x.
    sol = similar(P.a)
    chunks = initialize_chunks(length(sol); nchunks=nthreads)
    function b_callback(x, outiter)
        # CQK
        b = @benchmarkable cqk!($sol, $P, chunks=($chunks), x0=($x))
        time = estimatetime(b)
        iter, flag = cqk!(sol, P, nchunks=nthreads, x0=(x))
        infeas = abs(dot(P.b, sol) - P.r)   # P.r = 0
        push!(results,
             [instance, n, outiter, "cqk", nthreads, iter, flag, time, infeas])

        # CMS_CQN
        if nthreads == 1
            b = @benchmarkable cms_cqn!($sol, $P, x0=($x))
            time = estimatetime(b)
            iter, flag = cms_cqn!(sol, P, x0=(x))
            infeas = abs(dot(P.b, sol) - P.r)   # P.r=0
            push!(results,
                 [instance, n, outiter, "cqn", 1, iter, flag, time, infeas])
        end

        return false
    end

    # --------
    # CALL SPG
    # --------
    dualsol, spgiter, flag = spg(
        n,
        x -> f(x, H, Z, y),
        (g, x) -> g!(g, x, H, Z, y),
        (p, z, x0) -> proj!(p, z, x0, chunks, P),
        l = P.l, u = P.u,
        callback = isnothing(results) ? nothing : b_callback,
        maxiters = 500,
        verbose = verbose
    )

    # Revert variable changes
    dualsol[maskchg] .*= -1.0

    return dualsol, spgiter, flag
end

function svm_executed(results, instance, nthreads)
    if !isempty(
        results[
            (results.Instance .== instance) .& (results.threads .== nthreads),
            :
        ]
    )
        @printf(
            "%s with %d thread(s) already executed. Skipping...\n",
            instance, nthreads
        )
        return true
    end
    return false
end

function z_score!(Z, itrain)
    for f in 1:size(Z,1)
        m = mean(Z[f,itrain])
        std_dev = std(Z[f,itrain], mean=m)
        Z[f,itrain] .-= m
        Z[f,itrain] ./= std_dev
    end
end

# Tuning SVM parameters by a simple grid search with k-fold cross validation
function svm_tune(Z, w; k = 4)
    bestγ = 0.0
    bestC = 0.0

    ninst = size(Z, 2)

    # divide data in k balanced parts
    len, inc = divrem(ninst, k)
    itest = []
    start = 1
    @inbounds for i in 1:inc
        push!(itest, start:(start + len))
        start += len + 1
    end
    @inbounds for i in (inc + 1):k
        push!(itest, start:(start + len - 1))
        start += len
    end

    dualsol = Vector{Float64}(undef, ninst)
    besterror = Inf

    # Permutation to shuffle data (the same for all tests)
    perm = randperm(ninst)

    ZZ = similar(Z)
    ww = similar(w)

    # "log ranges" for γ and C
    γ0 = 1.0 / size(Z, 1)
    C0 = 1.0

    γs = Float64[]
    Cs = Float64[]

    for disp = -0.5:0.5:1.5
        push!(γs, 10^(log(10, γ0) + disp))
    end
    for disp = -1.0:0.5:1.0
        push!(Cs, 10^(log(10, C0) + disp))
    end

    for γ in γs, C in Cs
        # Kernel
        K(a,b) = exp(-γ * sqeuclidean(a,b))

        # "Lazy" Hessian
        H(Z) = @inbounds @views [ K(Z[:,i],Z[:,j]) for i=1:size(Z,2), j=1:size(Z,2) ]

        error_measure = 0.0

        # cross validation
        for t in itest
            # Instances indices for training
            itrain = setdiff(1:ninst, t)

            # Shuffle data according perm
            ZZ .= Z[:,perm]
            ww .= w[perm]

            # Scale features of training data (z_score)
            z_score!(ZZ, itrain)

            red_dualsol, _, flag = svm_solve(
                "tuning", ZZ[:,itrain], ww[itrain], 1; γ = γ, C = C, verbose = 0
            )

            if flag == :solved
                dualsol .= 0.0
                dualsol[itrain] .= red_dualsol

                # Compute b using all dual variables in (0,C)
                b = 0.0
                count_valid = 0
                for j in itrain
                    if (dualsol[j] > 1e-6) && (dualsol[j] < C - 1e-6)
                        bj = 0.0
                        for i in itrain
                            bj -= dualsol[i] * ww[i] * K(ZZ[:,i], ZZ[:,j])
                        end
                        b += ww[j] - bj
                        count_valid += 1
                    end
                end
                b /= count_valid

                for v in t
                    s = 0.0
                    for i in itrain
                        s += dualsol[i] * ww[i] * K(ZZ[:,i], ZZ[:,v])
                    end
                    s += b
                    if abs(ww[v] * s - 1) > 1e-4
                        error_measure += abs(ww[v] * s - 1)
                    end
                end
                if error_measure / length(itest) >= besterror
                    break
                end
            else
                error_measure = Inf
                break
            end
        end

        error_measure /= length(itest)

        if error_measure < besterror
            bestγ = γ
            bestC = C
            besterror = error_measure
        end
    end

    return bestγ, bestC
end

# Read a dataset
function svm_readdataset(id)
    d = datasets[datasets.id .== id,:]
    n = d.NumberOfInstances[1]
    m = d.NumberOfNumericFeatures[1]
    if (d.NumberOfSymbolicFeatures[1] > 1) || (m != d.NumberOfFeatures[1] - 1)
        println("$(d.name[1]): Invalid number of numeric features")
        return nothing, nothing
    end
    data = []
    try
        data = DataFrame(OpenML.load(id))
    catch
        println("$(d.name[1]): Error while parsing")
        return nothing, nothing
    end
    classes = unique(data[:,end])
    if length(classes) != 2
        return nothing, nothing
    end
    Z = Matrix{Float64}(undef, m, n)
    Z .= Float64.(Matrix(data[:, 1:(end-1)])')
    w = ones(n)
    # classes[1]: 1, classes[2]: -1
    w[data[:,end] .== classes[2]] .= -1
    return Z, w
end

function svm_merge_params()
    # Read main parameter file
    param = jld2_read("param", "svm_param.jld2")
    if isnothing(param)
        param = Dict()
    end

    # Search for partial parameter files and merge their content with 'param'
    files = readdir(".")
    for f in files
        if startswith(f, "svm_param_") && endswith(f, ".jld2")
            parami = jld2_read("param", f)
            param = merge(param, parami)
        end
    end
    try
        jldsave("svm_param.jld2"; param)

        # Delete partial files, only executed if svm_param.jld2 was updated
        for f in files
            if startswith(f, "svm_param_") && endswith(f, ".jld2")
                rm(f)
            end
        end
    catch
        println("Fail to save merged parameter file.")
    end
end

# Try to extract source of a dataset
function svm_dataset_source(id)
    source = ""
    # Dataset description in Markdown
    mdtext = OpenML.describe_dataset(id)
    # Search for the paragraph containing the word "Source"
    par = 0
    for i in 1:length(mdtext)
        if contains(string(mdtext.content[i]), "Source")
            par = i
            break
        end
    end
    if par > 0
        # Search within the paragraph
        for i in 1:(length(mdtext.content[par].content)-2)
            if contains(string(mdtext.content[par].content[i]), "Source")
                if typeof(mdtext.content[par].content[i+2]) == Markdown.Link
                    source = mdtext.content[par].content[i+2].text
                end
                break
            end
        end
    end
    return source
end

# Run tuning
function svm_alltune()
    # Merge all parameter files to collect old, possibly unfinished tests
    svm_merge_params()

    param_all = jld2_read("param", "svm_param.jld2")
    if isnothing(param_all)
        param_all = Dict()
    end

    # Tuning
    Threads.@threads for d in eachrow(datasets)
        param = jld2_read("param", "svm_param_$(Threads.threadid()).jld2")
        if isnothing(param)
            param = Dict()
        end

        if haskey(param_all, d.id)
            println("$(d.name): Already tuned. Skipping...")
            continue
        else
            if isempty(svm_dataset_source(d.id))
                continue
            end
            Z, w = svm_readdataset(d.id)
            if isnothing(Z)
                continue
            end

            # Tune parameters by a simple grid search
            println("$(d.name): Tuning parameters...")
            γ, C = svm_tune(Z, w)
            println("\n$(d.name): done. Parameters: γ = $(γ), C = $(C)")

            push!(param, d.id => [γ; C])

            # Save parameters file
            jldsave("svm_param_$(Threads.threadid()).jld2"; param)
        end
    end

    svm_merge_params()
end

function svm_alltests(cont)
    nthreads = Threads.nthreads()

    param = jld2_read("param", "svm_param.jld2")

    output = "results_svm.jld2"

    # Results
    results = jld2_read("results", output; test = cont)
    if isnothing(results)
        results = DataFrame(
            [
                "Instance" => String[]
                "n" => Int64[]
                "outiter" => Int64[]
                "Algorithm" => String[]
                "threads" => Int64[]
                "iter" => Int64[]
                "st" => []
                "time" => Float64[]
                "infeas" => Float64[]
            ]
        )
    end

    for id in keys(param)
        d = datasets[datasets.id .== id, :]

        if !svm_executed(results, d.name[1], nthreads)
            γ, C = param[id]
            if (γ == 0.0) || (C == 0.0)
                continue
            end

            n = d.NumberOfInstances[1]
            m = d.NumberOfNumericFeatures[1]
            println("\nDataset: $(d.name[1]) (id $(id))   Instances: $(n)   Features: $(m)")

            Z, w = svm_readdataset(id)
            if isnothing(Z)
                println('-'^98)
                continue
            end

            # Scale features
            z_score!(Z, 1:size(Z,2))

            _, _, flag = svm_solve(
                d.name[1], Z, w, nthreads;
                results = results, γ = γ, C = C
            )

            if flag != :solved
                println("SPG fails.")
                println('-'^98)
                continue
            end

            println('-'^98)

            # Save results
            jldsave(output; results)
        end
    end
end
