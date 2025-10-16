include("spg.jl")

# Objective function
function f(x, H, Z, y)
    return 0.5 * (x' * H(Z) * x) - sum(sign.(y) .* x)
end

# Gradient of objective function
function g!(g, x, H, Z, y)
    g .= (H(Z) * x) .- sign.(y)
end

# Projection (solve particular CQK)
function proj!(p, z, x0, P::CQKProblem)
    P.a .= z
    _, flag = cqk!(p, P, x0=(x0))
    if flag != :solved
        error("Error while solving CQK (exit status: $(flag))")
    end
end

function svm_solve(
    instance, Z, y, nthreads; results = nothing, sigma = 5.0, C = 1.0, verbose = 1
)
    @assert sigma > 0.0 throw(ArgumentError("sigma must be positive"))
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
    frac = 0.5 / sigma^2
    K(zi,zj) = exp(-frac * sqeuclidean(zi,zj))

    # "Lazy" Hessian
    H(Z) = @inbounds @views [ K(Z[:,i],Z[:,j]) for i=1:n, j=1:n ]

    tmp = similar(P.a)

    # Callback function for benchmarking
    # P is relative to x, which must be updated before by proj!. For benchmarking,
    # we start at x0 = x.
    sol = similar(P.a)
    function b_callback(x, outiter)
        # CQK
        chunks = initialize_chunks(length(sol); nchunks=nthreads)
        b = @benchmarkable cqk!($sol, $P, chunks=($chunks), x0=($x))
        time = estimatetime(b)
        iter, flag = cqk(P, nchunks=nthreads, x0=(x))[2:3]
        infeas = abs(dot(P.b, sol) - P.r)   # P.r = 0
        push!(results,
             [instance, n, outiter, "cqk", nthreads, iter, flag, time, infeas])

        # CMS_CQN
        if nthreads == 1
            b = @benchmarkable cms_cqn!($sol, $P, x0=($x))
            time = estimatetime(b)
            iter, flag = cms_cqn(P, x0=(x))[2:3]
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
        (p, z, x0) -> proj!(p, z, x0, P),
        l = P.l, u = P.u,
        callback = isnothing(results) ? nothing : b_callback,
        verbose = verbose
    )

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

# Tuning SVM parameters by a simple grid search w/ cross validation
function svm_tune(Z, w; k = 4)
    bestsigma = 0.0
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

    besterror = Inf

    for sigma = 2.0:1.0:10.0, C = 0.1:0.5:5.0
        # Kernel
        frac = 0.5 / sigma^2
        K(zi,zj) = exp(-frac * sqeuclidean(zi,zj))

        # "Lazy" Hessian
        H(Z) = @inbounds @views [ K(Z[:,i],Z[:,j]) for i=1:size(Z,2), j=1:size(Z,2) ]

        error_measure = 0.0

        # cross validation
        for t in itest
            itrain = setdiff(1:ninst, t)

            dualsol, _, flag = svm_solve(
                "tuning", Z[:,t], w[t], 1; sigma = sigma, C = C, verbose = 0
            )

            if flag != :solved
                #TODO: compute classification errors on testdata
                error_measure += 0.0
            end
        end

        error_measure /= length(itest)

        if error_measure < besterror
            bestsigma = sigma
            bestC = C
        end
    end

    return bestsigma, bestC
end

function svm_alltests(cont)
    nthreads = Threads.nthreads()

    param = Dict()

    if isfile("svm_param.jld2")
        jld2file = jldopen("svm_param.jld2", "r")
        param = read(jld2file, "param")
        close(jld2file)
    end

    output = "results_svm.jld2"

    # Results
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
    if cont && isfile(output)
        jld2file = jldopen(output, "r")
        results = read(jld2file, "results")
        close(jld2file)
    end

    # SVM
    # Filter datasets for binary classification from UCI
    datasets = OpenML.list_datasets(
        tag = "uci",
        filter="number_classes/2/number_instances/100..1000/number_missing_values/0",
        output_format = DataFrame
    )

    for d in eachrow(datasets)
        n = d.NumberOfNumericFeatures
        m = d.NumberOfInstances

        if (d.NumberOfSymbolicFeatures > 1) || (n != d.NumberOfFeatures - 1)
            continue
        end

        if !svm_executed(results, d.name, nthreads)
            println("\nDataset: $(d.name) (id $(d.id))    Num instances: $(m)    Num features: $(n)")

            Z = []
            w = []
            try
                data = DataFrame(OpenML.load(d.id))
                classes = unique(data[:,end])
                if length(classes) != 2
                    continue
                end
                Z = Float64.(Matrix(data[:, 1:(end-1)])')
                w = ones(m)
                # classes[1]: 1, classes[2]: -1
                w[data[:,end] .== classes[2]] .= -1
            catch
                println("Error while parsing")
                println('-'^98)
                continue
            end

#             try
                if haskey(param, d.name)
                    # Parameters already computed
                    sigma, C = param[d.name]
                else
                    # Tune parameters by a simple grid search
                    println("Tuning parameters for $(d.name)...")
                    sigma, C = svm_tune(Z, w)
                    println("done")
                    if (sigma == 0.0) || (C == 0.0)
                        # Tuning failed
                        println("Tuning failed. Skipping...")
                        continue
                    end
                    push!(param, d.name => [sigma; C])

                    # Update parameters file
                    jldsave("svm_param.jld2"; param)
                end

                svm_solve(
                    d.name, Z, w, nthreads;
                    results = results, sigma = sigma, C = C
                )
#             catch err
                println("Error while solving.\nMessage: $(err)")
                println('-'^98)
                continue
#             end

            println('-'^98)

            # Save results
            jldsave(output; results)
        end
    end
end
