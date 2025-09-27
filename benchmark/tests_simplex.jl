################################
# Auxiliary functions
################################

function rand_nonzerovec(D, n)
    x = rand(D, n)
    while any(x .== 0.0)
        x .= rand(D, n)
    end
    return x
end

function simplex_infeas(sol)
    return abs(sum(sol) - 1.0)
end

################################
# Instances
################################
SIMPLEX_INSTANCES = INSTANCE[]

simplex_names = ["Random 1", "Random 2", "Random 3"]
simplex_sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
simplex_gen = [
    n -> rand_nonzerovec(Uniform(0, 1), n),
    n -> rand_nonzerovec(Normal(0, 1), n),
    n -> rand_nonzerovec(Normal(0, 1e-3), n)
]

unitsize64 = sizeof(1.0)
maxn = maxsize(unitsize64, USECUDA)
for i in eachindex(simplex_names), n in simplex_sizes
    if n > maxn
        @info "Problem size $n may not fit in RAM or GPU memory, skipping."
        continue
    end
    push!(SIMPLEX_INSTANCES, INSTANCE(simplex_names[i], n, simplex_gen[i]))
end

##############
# Methods
##############
SIMPLEX_METHODS = METHOD[]

# Simplex, dense, CPU, Float64
push!(
    SIMPLEX_METHODS,
    METHOD(
        "simplex (CPU, FP64)",
        identity,
        (P, nthreads) -> b_dense(P, simplex_proj!, nthreads),
        (P, nthreads) -> simplex_proj(P; nchunks=nthreads)[2:3],
        (P, nthreads) -> simplex_infeas(simplex_proj(P; nchunks=nthreads)[1]),
        (P, nthreads) -> Inf
    )
)

# Simplex, sparse, CPU, Float64
push!(
    SIMPLEX_METHODS,
    METHOD(
        "sp simplex (CPU, FP64)",
        identity,
        (P, nthreads) -> b_sparse(P, spsimplex_proj, nthreads),
        (P, nthreads) -> spsimplex_proj(P; nchunks=nthreads)[2:3],
        (P, nthreads) -> simplex_infeas(spsimplex_proj(P; nchunks=nthreads)[1]),
        (P, nthreads) -> Inf
    )
)

# Condat C code, dense, CPU, Float64
push!(
    SIMPLEX_METHODS,
    METHOD(
        "Condat C",
        identity,
        (P, nthreads) -> b_Ccondat(P, nthreads),
        (P, nthreads) -> (-1, :solved),
        (P, nthreads) -> simplex_infeas(condat_proj(P)),
        (P, nthreads) -> Inf
    )
)

# Parallel Condat simplex, sparse, CPU, Float64
push!(
    SIMPLEX_METHODS,
    METHOD(
        "P Condat (simplex)",
        identity,
        (P, nthreads) -> b_Pcondat(P, condat_s, condat_p, nthreads),
        (P, nthreads) -> (-1, :solved),
        (P, nthreads) -> if (nthreads == 1)
            simplex_infeas(condat_s(P, 1.0))
        else
            simplex_infeas(condat_p(P, 1.0, nthreads, 0.001))
        end,
        (P, nthreads) -> Inf
    )
)

# Simplex, dense, GPU, Float32
if USECUDA > 32
    push!(
        SIMPLEX_METHODS,
        METHOD(
            "simplex (GPU, FP64)",
            P -> CPUtoGPU(P, Float64),
            (P, nthreads) -> b_cuda(P, simplex_proj!, nthreads),
            (P, nthreads) -> simplex_proj(P)[2:3],
            (P, nthreads) -> simplex_infeas(simplex_proj(P)[1]),
            (P, nthreads) -> reldiff_sol(P, simplex_proj)
        )
    )
end

# Simplex, dense, CPU, Float32
push!(
    SIMPLEX_METHODS,
    METHOD(
        "simplex (CPU, FP32)",
        F64toF32,
        (P, nthreads) -> b_dense(P, simplex_proj!, nthreads),
        (P, nthreads) -> simplex_proj(P; nchunks=nthreads)[2:3],
        (P, nthreads) -> simplex_infeas(simplex_proj(P; nchunks=nthreads)[1]),
        (P, nthreads) -> Inf
    )
)

# Simplex, dense, GPU, Float32
if USECUDA > 0
    push!(
        SIMPLEX_METHODS,
        METHOD(
            "simplex (GPU, FP32)",
            P -> CPUtoGPU(P, Float32),
            (P, nthreads) -> b_cuda(P, simplex_proj!, nthreads),
            (P, nthreads) -> simplex_proj(P)[2:3],
            (P, nthreads) -> simplex_infeas(simplex_proj(P)[1]),
            (P, nthreads) -> reldiff_sol(P, simplex_proj)
        )
    )
end
