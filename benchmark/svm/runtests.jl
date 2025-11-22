include("../common/common.jl")

using Distances

datasets_file = joipath(projectpath, "svm", "datasets.jld2")

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
    n,m = size(datasets[row].data)
    m -= 1
    Z = Matrix{Float64}(undef, m, n)
    Z .= Float64.(Matrix(datasets[row].data[:, 1:(end-1)])')
    w = ones(n)
    # c[1]: 1, c[2]: -1
    w[datasets[row].data[:,end] .== c[1]] .= -1
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
function f(x, H, sgny)
    return 0.5 * dot(x, Symmetric(H), x) - dot(sgny, x)
end

# Gradient of objective function
function g!(g, x, H, sgny)
    g .= Symmetric(H) * x .- sgny
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

# Compute classification error
function svm_error(dualsol, H, y, C)
    error_measure = 0.0
    nerror = 0

    n0 = 0
    nint = 0
    nC = 0

    # Compute b using all dual variables in (0,C)
    b = 0.0
    nint = 0
    for j in eachindex(y)
        if (dualsol[j] > eps()) && (dualsol[j] < C - eps())
            bj = 0.0
            for i in eachindex(y)
                bj -= dualsol[i] * y[i] * H[i,j]
            end
            b += y[j] - bj
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
        # If there is no variables in (0,C), use those equal to C
    end

    for j in eachindex(y)
        s = 0.0
        for i in eachindex(y)
            s += dualsol[i] * y[i] * H[i,j]
        end
        s += b
        errorj = abs(y[j] * s - 1)
        error_measure += errorj
        if errorj > 1e-4
            nerror += 1
        end
    end

    return error_measure, nerror
end

# Apply SPG and optionally benchmark
# Training data is considered scaled here!
function solve(
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

    Hx = Vector{Float64}(undef, n)

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
        maxiters = 300, x0 = x0, eps = 1e-5,
        verbose = verbose
    )

    # Revert variable changes
    dualsol[maskchg] .*= -1.0

    error_measure, nerror = svm_error(dualsol, Symmetric(H), y, C)

    return dualsol, spgiter, flag, error_measure, nerror
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
function alltests(cont; max_inst = 0)
    nthreads = Threads.nthreads()

    datasets = jld2_read("datasets", datasets_file)
    if isempty(datasets)
        return
    end

    output = joipath(projectpath, "results", "results_svm.jld2")

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

    for d in [4]
        if !executed(results, datasets[d].name, nthreads)
            γ = 1.0 / datasets[d].features
            C = 1.0

            n,m = size(datasets[d].data)
            m -= 1
            println("\nDataset: $(datasets[d].name) (id $(datasets[d].id))   Instances: $(n)   Features: $(m)")
            println("Parameters: γ = $(γ), C = $(C)")

            Z, w = readdataset(datasets[d].id, datasets)
            if isnothing(Z)
                println('-'^98)
                continue
            end

            # Select instances
#             if max_inst <= 0
#                 # Select 60% of the instances randomly
#                 max_inst = ceil(Int64, length(w) * 0.6)
#             end
#             mask_inst = rand(1:length(w), min(max_inst, length(w)))
#             println("Number of instances considered: $(length(mask_inst))")
#
#             # Filter data
#             Z = Z[:,mask_inst]
#             w = w[mask_inst]

            _, it, flag, error_measure, nerror = solve(
                datasets[d].name, Z, w, nthreads;
                γ = γ, C = C, verbose = 1,
#                 x0 = fill(1.0,n)
            )

            @show error_measure
            @show nerror

            if flag != :solved
                println("SPG fails.")
                println('-'^98)
                continue
            end

            # SPG iterations for benchmarking
#             nrange = 100
#             it_range = sort(union(1:min(it, nrange), max(1, it - nrange + 1):max(1, it)))
#             println("Benchmark iterations: Left range = 1:$(min(it, nrange)),  right range = $(max(1, it - nrange + 1)):$(max(1, it))")
#
#             # run again... perform benchmark for iterations in "it_range"
#             _, _, flag = solve(
#                 datasets[d].name, Z, w, nthreads;
#                 results = results, γ = γ, C = C, verbose = 0
#             )
#
#             println('-'^98)
#
#             # Save results
#             jldsave(output; results)
        end
    end
end

# Main function
function main(args)
    # Get command line parameters
    opts = get_parameters()

    println("===================\nSVM\n===================")
    random_alltests(opts["continue"])

    return 0
end

# Run main if non-iteractive
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
