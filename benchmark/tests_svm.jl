include("spg.jl")

function svm_solve(
    instance, Z, w, nthreads, results; sigma = 5.0, C = 1.0
)
    @assert sigma > 0.0 throw(ArgumentError("sigma must be positive"))
    @assert C > 0.0 throw(ArgumentError("C must be positive"))
    @assert !isempty(instance) throw(ArgumentError("instance must be provided"))

    n = size(Z,2)

    # CQK for subproblems
    # Note: P.b must be positive, so we change variables (P.l, P.u, P.a must be
    # adjusted)
    P = CQKProblem(ones(n), zeros(n), abs.(w), 0.0, zeros(n), fill(C,n))
    maskchg = (w .< 0.0)
    P.l[maskchg] .= -C
    P.u[maskchg] .= 0.0

    # Kernel
    frac = 0.5 / sigma^2
    K(zi,zj) = exp(frac * sqeuclidean(zi,zj))

    # "Lazy" Hessian
    H(Z) = @inbounds @views [ K(Z[:,i],Z[:,j]) for i=1:n, j=1:n ]

    # Objective function
    function f(x)
        return 0.5 * (x' * H(Z) * x) - sum(sign.(w) .* x)
    end

    # Gradient of objective function
    function g!(g, x)
        g .= (H(Z) * (sign.(w) .* x)) .- sign.(w)
    end

    # Direction (solve particular CQK)
    function d!(d, x, lambda, g)
        @. P.a = x - lambda * g
        _, flag = cqk!(d, P)
        if flag != :solved
            error("Error while solving CQK (lambda = $(lambda), exit status: $(flag))")
        end
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
    # P is relative to x, which must be updated before by d!. For benchmarking,
    # we start at x0 = xprev = "previous x". In the first outer iteration, x0 is
    # undefined.
    xprev = Float64[]
    sol = similar(P.a)
    function b_callback(x, outiter)
        # CQK
        chunks = initialize_chunks(length(sol); nchunks=nthreads)
        b = @benchmarkable cqk!($sol, $P, chunks=($chunks), x0=($xprev))
        time = estimatetime(b)
        iter, flag = cqk(P, nchunks=nthreads, x0=(xprev))[2:3]
        infeas = abs(dot(P.b, sol) - P.r)
        push!(results,
             [instance, n, outiter, "cqk", nthreads, iter, flag, time, infeas])

        # CMS_CQN
        if nthreads == 1
            b = @benchmarkable cms_cqn!($sol, $P)
            time = estimatetime(b)
            iter, flag = cms_cqn(P, x0=(xprev))[2:3]
            infeas = abs(dot(P.b, sol) - P.r)
            push!(results,
                 [instance, n, outiter, "cqn", 1, iter, flag, time, infeas])
        end

        # Update xprev for the next round
        isempty(xprev) ? xprev = deepcopy(x) : xprev .= x
    end

    # --------
    # CALL SPG
    # --------
    spg(n, f, g!, d!, pg_supnorm, l = P.l, u = P.u, callback = b_callback)

    return
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

function svm_alltests(cont)
    nthreads = Threads.nthreads()

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

    # IRIS
    if !svm_executed(results, "iris", nthreads)
        println("\nDataset: iris\n")
        data = UCIData.dataset("iris")
        Z = Matrix(data[1:100, 2:5])'
        w = ones(100)
        w[1:50] .= -1.0

        svm_solve("iris", Z, w, nthreads, results; sigma = 5.0, C = 1.0)

        # Save results
        jldsave(output; results)
    end
end
