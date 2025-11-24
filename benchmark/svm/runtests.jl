include("../common/common.jl")

using Distances
using OpenML

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

datasets_file = joinpath(projectpath, "svm", "datasets.jld2")

include("../common/spg.jl")
include("dataset.jl")

# Read a dataset from the list "datasets"
function readdataset(id, datasets)
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
    if length(c) != 2
        return nothing, nothing
    end
    n, m = size(datasets[row].data)
    m -= 1
    Z = Matrix{Float64}(undef, m, n)
    Z .= Float64.(Matrix(datasets[row].data[:, 1:(end-1)])')
    y = ones(n)
    # c[1]: 1, c[2]: -1
    y[datasets[row].data[:,end] .== c[1]] .= -1
    return Z, y
end

# Given γ and data Z, creates the nxn dense Hessian
function denseH(Z, γ)
    H = pairwise(SqEuclidean(), Z, Z, dims=2)
    H .= exp.(-γ .* H)
    return H
end

# Objective function
function f(x, H, sgny)
    return 0.5 * dot(x, Symmetric(H), x) - dot(sgny, x)
end

# Gradient of objective function
function g!(g, x, H, sgny)
    mul!(g, Symmetric(H), x)
    g .-= sgny
end

# Projection (solve particular CQK), without warm start
# p is the solution
# z is copied to P.a for later benchmarking. This is the vector x - lambda*g
# returned by SPG. To maintain the same SPG sequence, we use the sequential
# algorithm
function proj!(p, z, P::CQKProblem)
    P.a .= z
    _, flag = cqk!(p, P, nchunks=1, maxiters=500)
    return (flag == :solved)
end

# Classification
function classification(dualsol, H, y, C)
    error_measure = 0.0
    nerror = 0

    n0 = 0
    nint = 0
    nC = 0

    # Compute b using all dual variables in (0,C)
    b = 0.0
    nint = 0
    @inbounds for j in eachindex(y)
        if (dualsol[j] > eps()) && (dualsol[j] < C - eps())
            bj = 0.0
            for i in eachindex(y)
                bj += dualsol[i] * y[i] * H[i,j]
            end
            b += bj - y[j]
            nint += 1
        elseif dualsol[j] <= eps()
            n0 += 1
        else
            nC += 1
        end
    end

    if nint > 0
        b /= nint
    else
        # there is no variables in (0,C)
        L = -Inf
        U = Inf
        @inbounds for j in eachindex(y)
            s = 0.0
            for i in eachindex(y)
                s += dualsol[i] * y[i] * H[i,j]
            end
            if dualsol[j] <= eps()
                if y[j] > 0.0
                    L = max(L, s - 1.0)
                else
                    U = min(U, s + 1.0)
                end
            else
                if y[j] > 0.0
                    U = min(U, s - 1.0)
                else
                    L = max(L, s + 1.0)
                end
            end
        end
        b = (L + U)/2
    end

    @inbounds for j in eachindex(y)
        s = 0.0
        for i in eachindex(y)
            s += dualsol[i] * y[i] * H[i,j]
        end
        s -= b
        if signbit(s) != signbit(y[j])
            nerror += 1
        end
    end

    return nerror, n0, nint, nC
end

# Apply SPG and optionally benchmark
# Training data is considered scaled here!
function solve(
    instance, Z, y, nthreads;
    results = nothing,
    γ = 0.01,
    C = 1.0,
    x0 = Float64[],
    brange = 1:10^10,
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
        H = denseH(Z, γ)
    catch
        @error "Error while computing H"
        return 0, :H_error, 0, 0, 0, 0, 0
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
        x -> f(x, H, sgny),
        (g, x) -> g!(g, x, H, sgny),
        (p, z, x0) -> proj!(p, z, P),
        l = P.l, u = P.u,
        callback = isnothing(results) ? nothing : b_callback,
        maxiters = 10^5, x0 = x0, eps = 1e-4,
        verbose = verbose
    )

    # Revert variable changes
    dualsol[maskchg] .*= -1.0

    # Classification quality
    nerr, n0, nint, nC = classification(dualsol, Symmetric(H), y, C)

    return spgiter, flag, nerr, n0, nint, nC
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

# All SVM tests
function alltests(cont)
    nthreads = Threads.nthreads()

    datasets = jld2_read("datasets", datasets_file)
    if isempty(datasets)
        return
    end

    output = joinpath(projectpath, "results", "results_svm.jld2")

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

    for d in eachindex(datasets)
        if !executed(results, datasets[d].name, nthreads)
            # Params for MNIST / cdc_diabetes
            γ = (d == 1) ? 0.007 : 0.5
            C = (d == 1) ? 5.0 : 10.0

            n, m = size(datasets[d].data)
            m -= 1
            println("\nDataset: $(datasets[d].name) (id $(datasets[d].id))")
            println("Instances: $(n)")
            println("Features: $(m)")
            @printf("Parameters: γ = %12.8lf, C = %12.8lf\n", γ, C)

            Z, y = readdataset(datasets[d].id, datasets)
            if isnothing(Z)
                println('-'^98)
                continue
            end

            it, flag, nerror, n0, nint, nC = solve(
                datasets[d].name, Z, y, nthreads;
                γ = γ, C = C, verbose = 0
            )

            if flag != :solved
                println('-'^98)
                continue
            end

            println("# dual variables = 0: $(n0)")
            @printf("# dual variables in (0,C): %d (%.3lf %%)", nint, 100 * (nint/n))
            println("# dual variables = C: $(nC)")
            @printf("Error: %d samples (%.3lf %%)", nerror, 100 * (nerror/n))

            # SPG iterations for benchmarking
            nrange = 100
            it_range = sort(union(1:min(it, nrange), max(1, it - nrange + 1):max(1, it)))
            println("Benchmark iterations: 1:$(min(it, nrange)),  $(max(1, it - nrange + 1)):$(max(1, it))")

            # run again... perform benchmark for iterations in "it_range"
            _, _, flag = solve(
                datasets[d].name, Z, y, nthreads;
                results = results, γ = γ, C = C, verbose = 0
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

    println("===================\nSVM\n===================")
    alltests(opts["continue"])

    return 0
end

# Run main if non-iteractive
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
