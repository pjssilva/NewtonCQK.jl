# Filter datasets for binary classification
matrices_path = "SC_matrices"
matrices = readdir(matrices_path)

# Objective function
function bp_f(x, A, rhs)
    Axrhs = @views A*x .- rhs
    return 0.5 * dot(Axrhs, Axrhs)
end

# Gradient of objective function
function bp_g!(g, x, A, rhs)
    g .= A' * (A*x .- rhs)
end

# Projection (solve particular l1-ball projection problem)
function bp_proj!(p, z, x0, chunks, y, r)
    y .= z
    _, flag = l1ball_proj!(p, y, r=(r), chunks=(chunks))
    return (flag == :solved)
end

function bp_read_matrix(filename)
    A = []
    if isfile(filename)
        file = matopen(filename)
        A = read(file, "A")
        close(file)
    end
    return A
end


# Apply SPG and optionally benchmark
# Training data is considered scaled here!
function bp_solve(
    instance, A, rhs, nthreads;
    results = nothing, r = 1.0, verbose = 1
)
    @assert r > 0 throw(ArgumentError("r must be positive"))
    @assert !isempty(instance) throw(ArgumentError("instance must be provided"))
    @assert size(A,1) == length(rhs) throw(DimensionMismatch("A and b have incompatible dimensions"))

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
        # sparse l1ball
        b = @benchmarkable spl1ball_proj($y, r=($r), chunks=($chunks))
        time = estimatetime(b)
        sol, iter, flag = spl1ball_proj(y, r=(r), chunks=(chunks))
        infeas = max(0.0, sum(abs.(sol)) - r)
        push!(
            results,
            [instance, n, outiter, "Our algorithm without x0", nthreads, iter, flag, time, infeas]
        )

        b = @benchmarkable spl1ball_proj($y, r=($r), chunks=($chunks), x0=($x))
        time = estimatetime(b)
        sol, iter, flag = spl1ball_proj(y, r=(r), chunks=(chunks), x0=(x))
        infeas = max(0.0, sum(abs.(sol)) - r)
        push!(
            results,
            [instance, n, outiter, "Our algorithm 1 with x0", nthreads, iter, flag, time, infeas]
        )

        # Dai and Chen's algorithm
        if nthreads == 1
            b = @benchmarkable l1ball_condat_s($y, $r)
        else
            b = @benchmarkable l1ball_condat_p($y, $r, $nthreads, 0.001)
        end
        time = estimatetime(b)
        if nthreads == 1
            sol = l1ball_condat_s(y, r)
        else
            sol = l1ball_condat_p(y, r, nthreads, 0.001)
        end
        infeas = max(0.0, sum(abs.(sol)) - r)
        push!(
            results,
            [instance, n, outiter, "Dai and Chen's algorithm", nthreads, -1, :solved, time, infeas]
        )

        return false
    end

    # --------
    # CALL SPG
    # --------
    spgsol, spgiter, flag = spg(
        n,
        x -> bp_f(x, A, rhs),
        (g, x) -> bp_g!(g, x, A, rhs),
        (p, z, x0) -> bp_proj!(p, z, x0, chunks, y, r),
        l = -Inf, u = Inf,
        callback = isnothing(results) ? nothing : b_callback,
        maxiters = 500,
        verbose = verbose
    )

    return spgsol, spgiter, flag
end

function bp_executed(results, instance, nthreads)
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

function bp_alltests(cont)
    nthreads = Threads.nthreads()

    output = "results_bp.jld2"

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
        if !bp_executed(results, mat, nthreads)
            A = bp_read_matrix(joinpath(matrices_path, mat))
            if isempty(A)
                println("Error while reading $(mat)")
                continue
            end
            n = size(A,2)
            sparsex = zeros(n)
            nsparse = ceil(Int64, 0.05*n))
            sparsex[rand(1:n, nsparse)] .= 1.0
            rhs = A * sparsex

            _, _, flag = bp_solve(
                mat, A, rhs, nthreads;
                results = results, r = nsparse/10)
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
