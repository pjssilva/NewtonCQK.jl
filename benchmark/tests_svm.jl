struct DATASET
    id::Int64
    name::String
    features::Int64
    instances::Int64
    data::DataFrame
end

# Download selected datasets with OpenML
# In case of failure, you can run again to continue
function svm_load_datasets(; onlyshows=false, namecontains="", id=0)
    list = []
    try
        if id == 0
            list = OpenML.list_datasets(
                filter="number_classes/2/number_instances/25000..35000/number_missing_values/0",
                output_format = DataFrame
            )
        else
            list = OpenML.list_datasets(filter="data_id/$(id)", output_format = DataFrame)
        end
    catch
        println("Error while list datasets with OpenML.")
        println("This problem occurs probably due when try to access the server. Try again.")
        return
    end

    datasets = jld2_read("datasets", "datasets.jld2")
    if isnothing(datasets)
        datasets = DATASET[]
    end
    for d in eachrow(list)
        if contains(d.name, "seed")
            continue
        end
        if !contains(d.name, namecontains)
            continue
        end
        if (d.NumberOfSymbolicFeatures > 1) || (d.NumberOfNumericFeatures != d.NumberOfFeatures - 1)
            continue
        end
        if onlyshows
            println("Dataset $(d.name), id $(d.id)")
            continue
        end
        alreadydownloaded = false
        for s in datasets
            if s.id == d.id
                alreadydownloaded = true
                break
            end
        end
        if alreadydownloaded
            continue
        end
        data = DataFrame()
        try
            data = DataFrame(OpenML.load(d.id))
        catch
            println("Fail to load dataset $(d.name), id $(d.id)")
            continue
        end
        push!(datasets, DATASET(d.id, d.name, d.NumberOfNumericFeatures, d.NumberOfInstances, data))
    end
    jldsave("datasets.jld2"; datasets)
end

# Read a dataset from the list "datasets"
# For nonbinary datasets, "class" must be provided
function svm_readdataset(id, datasets; class = nothing)
    if isempty(datasets)
        return nothing, nothing
    end
    row = 1
    while row <= length(datasets)
        if datasets[row].id == id
            break
        end
        row += 1
    end
    # Classes in data
    c = unique(datasets[row].data[:,end])
    if isnothing(class) && (length(c) != 2)
        return nothing, nothing
    end
    n,m = size(datasets[row].data)
    m -= 1
    Z = Matrix{Float64}(undef, m, n)
    Z .= Float64.(Matrix(datasets[row].data[:, 1:(end-1)])')
    w = ones(n)
    if isnothing(class)
        # c[1]: 1, c[2]: -1
        w[datasets[row].data[:,end] .== c[1]] .= -1
    else
        # class: -1, other: 1
        w[string.(datasets[row].data[:,end]) .== string(class)] .= -1
    end
    return Z, w
end

# Given γ and data Z, creates the nxn sparse Hessian
# discarding entries <= tol. Only the upper triangle is stored
function sparseH(Z, γ; tol = 1e-20)
    n = size(Z,2)
    I = Int32[]
    J = Int32[]
    V = Float64[]
    @inbounds for i in 1:(n-1)
        for j in (i+1):n
            # Gaussian kernel
            @views aux = exp(-γ * sqeuclidean(Z[:,i], Z[:,j]))
            if aux >= tol
                push!(I, i)
                push!(J, j)
                push!(V, aux)
            end
        end
    end
    return sparse(I, J, V, n, n)
end

# Objective function
function svm_f(x, H, sgny)
    return 0.5 * dot(x, Symmetric(H), x) - dot(sgny, x)
end

# Gradient of objective function
function svm_g!(g, x, H, sgny)
    g .= Symmetric(H) * x .- sgny
end

# Projection (solve particular CQK), without warm start
# p is the solution
# z is copied to P.a for later benchmarking. This is the vector x - lambda*g
# returned by SPG. To maintain the same SPG sequence, we only the sequential
# algorithm is applied
function svm_proj!(p, z, P::CQKProblem)
    P.a .= z
    _, flag = cqk!(p, P, nchunks=1, maxiters=500)
    return (flag == :solved)
end

# Apply SPG and optionally benchmark
# Training data is considered scaled here!
function svm_solve(
    instance, Z, y, nthreads;
    results = nothing,
    γ = 0.01,
    C = 1.0,
    x0 = Float64[],
    brange = 1:100_000_000,
    verbose = 1
)
    @assert γ > 0.0 throw(ArgumentError("γ must be positive"))
    @assert C > 0.0 throw(ArgumentError("C must be positive"))
    @assert !isempty(instance) throw(ArgumentError("instance must be provided"))
    @assert size(Z,2) == length(y) throw(DimensionMismatch("Z and y have incompatible dimensions"))

    n = size(Z,2)

    # Hessian
    H = []
    try
        H = sparseH(Z, γ)
    catch
        @error "Error when computing H"
        return [], 0, :H_error
    end

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

    # Callback function for benchmarking
    # P is relative to x, which must be updated before by proj!. For benchmarking,
    # we start at x0 = x.
    sol = similar(P.a)
    chunks = initialize_chunks(n; nchunks=nthreads)
    function b_callback(x, x0, outiter)
        if !(outiter in brange)
            return false
        end

        # CQK with without x0
        b = @benchmarkable cqk!($sol, $P, chunks=($chunks))
        time = estimatetime(b)
        iter, flag = cqk!(sol, P, chunks=(chunks))
        infeas = abs(dot(P.b, sol) - P.r)   # P.r = 0
        nfixed = count((sol .== P.l) .| (sol .== P.u))
        push!(
            results,
            [
                instance, n, outiter, "cqk (SVM)", nthreads,
                iter, flag, time, infeas,
                # we dot not store f and |g| as in basis pursuit
                nfixed, 0.0, 0.0
            ]
        )

        # CQK with x0
        b = @benchmarkable cqk!($sol, $P, chunks=($chunks), x0=($x0))
        time = estimatetime(b)
        iter, flag = cqk!(sol, P, chunks=(chunks), x0=(x0))
        infeas = abs(dot(P.b, sol) - P.r)   # P.r = 0
        nfixed = count((sol .== P.l) .| (sol .== P.u))
        push!(
            results,
            [
                instance, n, outiter, "cqk (SVM) x0", nthreads,
                iter, flag, time, infeas,
                nfixed, 0.0, 0.0
            ]
        )

        # CMS_CQN with x0
        if nthreads == 1
            b = @benchmarkable cms_cqn!($sol, $P, x0=($x0))
            time = estimatetime(b)
            iter, flag = cms_cqn!(sol, P, x0=(x0))
            infeas = abs(dot(P.b, sol) - P.r)   # P.r=0
            nfixed = count((sol .== P.l) .| (sol .== P.u))
            push!(
                results,
                [
                    instance, n, outiter, "cqn (SVM)", nthreads,
                    iter, flag, time, infeas,
                    nfixed, 0.0, 0.0
                ]
            )
        end

        # return false => SPG continues
        return false
    end

    # --------
    # CALL SPG
    # --------
    sgny = sign.(y)
    dualsol, spgiter, flag = spg(
        n,
        x -> svm_f(x, H, sgny),
        (g, x) -> svm_g!(g, x, H, sgny),
        (p, z, x0) -> svm_proj!(p, z, P),
        l = P.l, u = P.u,
        callback = isnothing(results) ? nothing : b_callback,
        maxiters = 300, x0 = x0, eps = 1e-5,
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

# All SVM tests
function svm_alltests(cont; max_inst = 0)
    datasets = jld2_read("datasets", "datasets.jld2")
    if isempty(datasets)
        return
    end
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
                "nonzeros" => Float64[]
                "f" => Float64[]
                "gsupn" => Float64[]
            ]
        )
    end

    for d in eachindex(datasets)
        if !svm_executed(results, datasets[d].name, nthreads)
            γ = 1.0 / datasets[d].features
            C = 1.0

            n,m = size(datasets[d].data)
            m -= 1
            println("\nDataset: $(datasets[d].name) (id $(datasets[d].id))   Instances: $(n)   Features: $(m)")
            println("Parameters: γ = $(γ), C = $(C)")

            Z, w = svm_readdataset(datasets[d].id, datasets)
            if isnothing(Z)
                println('-'^98)
                continue
            end

            # Select instances
            if max_inst <= 0
                # Select 60% of the instances randomly
                max_inst = ceil(Int64, length(w) * 0.6)
            end
            mask_inst = rand(1:length(w), min(max_inst, length(w)))
            println("Number of instances considered: $(length(mask_inst))")

            # Filter data
            Z = Z[:,mask_inst]
            w = w[mask_inst]

            _, it, flag = svm_solve(
                datasets[d].name, Z, w, nthreads;
                γ = γ, C = C, verbose = 0
            )

            if flag != :solved
                println("SPG fails.")
                println('-'^98)
                continue
            end

            # SPG iterations for benchmarking
            nrange = 100
            it_range = sort(union(1:min(it, nrange), max(1, it - nrange + 1):max(1, it)))
            println("Benchmark iterations: Left range = 1:$(min(it, nrange)),  right range = $(max(1, it - nrange + 1)):$(max(1, it))")

            # run again... perform benchmark for iterations in "it_range"
            _, _, flag = svm_solve(
                datasets[d].name, Z, w, nthreads;
                results = results, γ = γ, C = C, verbose = 0
            )

            println('-'^98)

            # Save results
            jldsave(output; results)
        end
    end
end
