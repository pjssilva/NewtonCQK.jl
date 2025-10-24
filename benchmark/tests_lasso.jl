# Filter datasets for binary classification
matrices = [
    "kneser_10_4_1",
    "wheel_601",
    "LargeRegFile",
    "GL7d21",
    "Hardesty3"
]

# Objective function
function f(x, A, b)
    Axb = @views A*x .- b
    return 0.5 * dot(Axb, Axb)
end

# Gradient of objective function
function g!(g, x, A, b)
    g .= A' * (A*x .- b)
end

# Projection (solve particular l1-ball projection problem)
function proj!(p, z, x0, chunks, y, r)
    y .= z
    _, flag = l1ball_proj!(p, y, r=(r), chunks=(chunks), x0=(x0))
    return (flag == :solved)
end

# Apply SPG and optionally benchmark
# Training data is considered scaled here!
function lasso_solve(
    instance, A, b, nthreads;
    results = nothing, r = 1.0, verbose = 1
)
    @assert r > 0 throw(ArgumentError("r must be positive"))
    @assert !isempty(instance) throw(ArgumentError("instance must be provided"))
    @assert size(A,1) == length(b) throw(DimensionMismatch("A and b have incompatible dimensions"))

    n = size(A,2)

    # The problem is
    # min  0.5*|Ax - b|^2
    # s.t. sum_i |x_i| <= r

    # Store x - lambda * g computed by SPG, the vector that needs to be projected
    y = Vector{Float64}(undef, n)

    # Callback function for benchmarking. We start at x0 = x.
    sol = similar(y)
    chunks = initialize_chunks(n; nchunks=nthreads)
    function b_callback(x, outiter)
        # CQK
        b = @benchmarkable spl1ball_proj($y, r=($r), chunks=($chunks), x0=($x))
        time = estimatetime(b)
        sol, iter, flag = spl1ball_proj(y, r=(r), nchunks=nthreads, x0=(x))
        infeas = max(0.0, sum(abs.(sol)) - r)
        push!(
            results,
            [instance, n, outiter, "spl1ball_proj", nthreads, iter, flag, time, infeas]
        )

        # Dai and Chen's algorithm
        if nthreads == 1
            b = @benchmarkable l1ball_condat_s($y, $r)
        else
            b = @benchmarkable l1ball_condat_p($y, $r, $nthreads, 0.001)
        end
        time = estimatetime(b)
        if nthreads == 1
            sol, iter, flag = l1ball_condat_s(y, r)
        else
            sol, iter, flag = l1ball_condat_p(y, r, nthreads, 0.001)
        end
        infeas = max(0.0, sum(abs.(sol)) - r)
        push!(
            results,
            [instance, n, outiter, "P Condat (l1ball)", nthreads, iter, flag, time, infeas]
        )

        return false
    end

    # --------
    # CALL SPG
    # --------
    spgsol, spgiter, flag = spg(
        n,
        x -> f(x, A, b),
        (g, x) -> g!(g, x, A, b),
        (p, z, x0) -> proj!(p, z, x0, chunks, y, r),
        l = -Inf, u = Inf,
        callback = isnothing(results) ? nothing : b_callback,
        maxiters = 500,
        verbose = verbose
    )

    return spgsol, spgiter, flag
end

function lasso_executed(results, instance, nthreads)
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

function lasso_alltests(cont)
    nthreads = Threads.nthreads()

    output = "results_lasso.jld2"

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

    for mat in matrices
        if !lasso_executed(results, mat, nthreads)
            A = matrixdepot("*/"*mat)
            b = A*ones(size(A,2))

            _, _, flag = lasso_solve(
                mat, A, b, nthreads;
                results = results, r = ceil(size(A,1)/100)
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
