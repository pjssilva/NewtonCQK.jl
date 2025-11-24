include("../common/common.jl")

using MAT

function get_parameters()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--continue"
        arg_type = Bool
        default = true
        help = "continue previous tests?"
    end
    return parse_args(s)
end

BLAS_NTHREADS = BLAS.get_num_threads()
BLAS.set_num_threads(1)

include("../common/spg.jl")

# Objective function
function f(x, A, b)
    Axb = @views A*x .- b
    return 0.5 * dot(Axb, Axb)
end

# Gradient of objective function
function g!(g, x, A, b)
    g .= A' * (A*x .- b)
end

# Project z onto the l1-ball with radius r, without warm start.
# p is the solution
# z is copied to y for later benchmarking. This is the vector x - lambda*g
# returned by SPG. To maintain the same SPG sequence, we only the sequential
# algorithm is applied
function proj!(p, z, y, r)
    y .= z
    _, flag = l1ball_proj!(p, y, r=(r), nchunks=1)
    return (flag == :solved)
end

# read problem in Matlab form
function read_problem(filename)
    A = []
    b = []
    if isfile(filename)
        file = matopen(filename)
        A = read(file, "A")
        b = read(file, "b")[:]
        close(file)
    end
    return A, b
end

# Apply SPG and optionally benchmark
function solve(
    instance, A, b, nthreads;
    results = nothing,
    r = 1.0,
    x0 = Float64[],
    brange = 1:10^10,
    verbose = 1
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
    g = similar(y)

    # Callback function for benchmarking
    chunks = initialize_chunks(n; nchunks=nthreads)
    function b_callback(x, x0, outiter)
        if !(outiter in brange)
            return false
        end

        # sparse l1ball without x0
        bm = @benchmarkable spl1ball_proj($y, r=($r), chunks=($chunks))
        time = estimatetime(bm)
        sol, iter, flag = spl1ball_proj(y, r=(r), chunks=(chunks))
        infeas = max(0.0, sum(abs.(sol)) - r)
        g!(g, sol, A, b)
        push!(
            results,
            [
                instance, n, outiter, "l1ball (bp)", nthreads,
                iter, flag, time, infeas,
                nnz(sol), f(sol, A, b), norm(g, Inf)
            ]
        )

        # sparse l1ball with x0
        bm = @benchmarkable spl1ball_proj($y, r=($r), chunks=($chunks), x0=($x0))
        time = estimatetime(bm)
        sol, iter, flag = spl1ball_proj(y, r=(r), chunks=(chunks), x0=(x0))
        infeas = max(0.0, sum(abs.(sol)) - r)
        g!(g, sol, A, b)
        push!(
            results,
            [
                instance, n, outiter, "l1ball (bp) x0", nthreads,
                iter, flag, time, infeas,
                nnz(sol), f(sol, A, b), norm(g, Inf)
            ]
        )

        # Dai and Chen's algorithm
        if nthreads == 1
            bm = @benchmarkable l1ball_condat_s($y, $r)
        else
            bm = @benchmarkable l1ball_condat_p($y, $r, $nthreads, 0.001)
        end
        time = estimatetime(bm)
        if nthreads == 1
            sol = l1ball_condat_s(y, r)
        else
            sol = l1ball_condat_p(y, r, nthreads, 0.001)
        end
        infeas = max(0.0, sum(abs.(sol)) - r)
        g!(g, sol, A, b)
        push!(
            results,
            [
                instance, n, outiter, "P Condat (l1ball)", nthreads,
                -1, :not_specified, time, infeas,
                nnz(sol), f(sol, A, b), norm(g, Inf)
            ]
        )

        # return false => SPG continues
        return false
    end

    # --------
    # CALL SPG
    # --------
    spgsol, spgiter, flag = spg(
        n,
        x -> f(x, A, b),
        (g, x) -> g!(g, x, A, b),
        (p, z, x0) -> proj!(p, z, y, r),
        l = -Inf, u = Inf,
        callback = isnothing(results) ? nothing : b_callback,
        maxiters = 10^5, x0 = x0, eps = 1e-4,
        verbose = verbose
    )

    return spgsol, spgiter, flag
end

function executed(results, instance, nthreads)
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

# All basis pursuit tests
function alltests(cont)
    nthreads = Threads.nthreads()

    # Matrices
    matrices_path = joinpath(projectpath, "basis_pursuit", "SC_matrices")
    matrices = ["SClog1.mat"; "SClog11.mat"]

    output = joinpath(projectpath, "results/results_basis_pursuit.jld2")

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
                "nfixed" => Float64[]
                "f" => Float64[]
                "gsupn" => Float64[]
            ]
        )
    end

    for mat in matrices
        if !executed(results, mat, nthreads)
            println("Instance $(mat), threads = $(nthreads)")

            # read matrix
            A, b = read_problem(joinpath(matrices_path, mat))
            if isempty(A) || isempty(b)
                println("Error while reading $(mat)")
                continue
            end

            if mat == "SClog11.mat"
                r = 35
            elseif mat == "SClog1.mat"
                r = 6700
            else
                r = size(A,2)/10
            end

            # count iterations, no benchmark
            x, it, flag = solve(
                mat, A, b, nthreads;
                r = r,
                verbose = 0
            )

            nonzeros = count(x .!= 0.0)
            @printf("# nonzeros = %d (%.3lf %%)\n", nonzeros, 100 * nonzeros/size(A,2))

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
            _, _, flag = solve(
                mat, A, b, nthreads;
                results = results,
                r = r, brange = it_range,
                verbose = 0
            )

            println('-'^98)

            # Save results
            jldsave(output; results)
        end
    end
end

# Main function
function main(args)
    # Get command line parameters
    opts = get_parameters()

    println("===================\nBasis pursuit\n===================")
    alltests(opts["continue"])

    return 0
end

# Run main if non-iteractive
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
