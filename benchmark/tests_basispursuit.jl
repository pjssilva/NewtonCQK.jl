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
    _, flag = l1ball_proj!(p, y, r=(r), nchunks=1)
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
    results = nothing,
    r = 1.0,
    x0 = Float64[],
    brange = 1:100000000,
    verbose = 1
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
    g = similar(y)

    xs_to_proj = SparseVector{Float64, Int64}[]
    x0s = SparseVector{Float64, Int64}[]

    # Callback function for benchmarking
    chunks = initialize_chunks(n; nchunks=nthreads)
    function b_callback(x, x0, outiter)
        if !(outiter in brange)
            return false
        end

        # sparse l1ball
        b = @benchmarkable spl1ball_proj($y, r=($r), chunks=($chunks))
        time = estimatetime(b)
        sol, iter, flag = spl1ball_proj(y, r=(r), chunks=(chunks))
        infeas = max(0.0, sum(abs.(sol)) - r)
        bp_g!(g, sol, A, rhs)
        push!(
            results,
            [
                instance, n, outiter, "l1ball (bp)", nthreads,
                iter, flag, time, infeas,
                nnz(sol), bp_f(sol, A, rhs), norm(g, Inf)
            ]
        )

        b = @benchmarkable spl1ball_proj($y, r=($r), chunks=($chunks), x0=($x0))
        time = estimatetime(b)
        sol, iter, flag = spl1ball_proj(y, r=(r), chunks=(chunks), x0=(x0))
        infeas = max(0.0, sum(abs.(sol)) - r)
        bp_g!(g, sol, A, rhs)
        push!(
            results,
            [
                instance, n, outiter, "l1ball (bp) x0", nthreads,
                iter, flag, time, infeas,
                nnz(sol), bp_f(sol, A, rhs), norm(g, Inf)
            ]
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
        bp_g!(g, sol, A, rhs)
        push!(
            results,
            [
                instance, n, outiter, "P Condat (l1ball)", nthreads,
                -1, :not_specified, time, infeas,
                nnz(sol), bp_f(sol, A, rhs), norm(g, Inf)
            ]
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
        maxiters = 50000, x0 = x0, eps = 1e-4,
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
                "nonzeros" => Float64[]
                "f" => Float64[]
                "gsupn" => Float64[]
            ]
        )
    end

    for mat in matrices
        if !bp_executed(results, mat, nthreads)
            println("Instance $(mat), threads = $(nthreads)")

            A = bp_read_matrix(joinpath(matrices_path, mat))
            if isempty(A)
                println("Error while reading $(mat)")
                continue
            end
            n = size(A,2)
            sparsex = zeros(n)
            nsparse = ceil(Int64, 0.05*n)
            sparsex[randperm(n)[1:nsparse]] .= 1.0
            rhs = A * sparsex
            println("n = $(n), nsparse = $(nsparse)")

            # count iterations, no benchmark
            x, it, flag = bp_solve(
                mat, A, rhs, nthreads;
                r = nsparse/4,
                verbose = 0
            )

            nonzeros = count(x .!= 0.0)
            println("SPG exit: it = $(it)  flag = $(flag)  f = $(bp_f(x, A, rhs))  #nonzeros = $(nonzeros) ($(nonzeros/n*100))%")

            if flag != :solved
                println("SPG fails.")
                println('-'^98)
                continue
            end

            # benchmark range
            nrange = 100
            it_range = sort(union(1:min(it, nrange), max(1, it - nrange + 1):max(1, it)))
            println("Benchmark iterations: Left range = 1:$(min(it, nrange)),  right range = $(max(1, it - nrange + 1)):$(max(1, it))")

            # run again, benchmark iterations in "it_range"
            _, _, flag = bp_solve(
                mat, A, rhs, nthreads;
                results = results,
                r = nsparse/4, brange = it_range,
                verbose = 0
            )

            println('-'^98)

            # Save results
            jldsave(output; results)
        end
    end
end
