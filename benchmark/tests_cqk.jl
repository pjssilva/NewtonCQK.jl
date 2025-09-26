#################################
# Test classes and aux functions
#################################

function CQK_uncorr(n)
    b = rand(Uniform(10,25), n)
    d = rand(Uniform(10,25), n)
    a = rand(Uniform(10,25), n)
    r1 = rand(Uniform(10,25), n)
    r2 = rand(Uniform(10,25), n)
    l = min.(r1, r2)
    u = max.(r1, r2)
    r = rand(Uniform(dot(b,l), dot(b,u)))
    return CQKProblem(d, a, b, r, l, u)
end

function CQK_wcorr(n)
    b = rand(Uniform(10,25), n)
    d = similar(b)
    a = similar(b)
    @. d = rand(Uniform(b - 5.0, b + 5.0))
    @. a = rand(Uniform(b - 5.0, b + 5.0))
    r1 = rand(Uniform(10,25), n)
    r2 = rand(Uniform(10,25), n)
    l = min.(r1, r2)
    u = max.(r1, r2)
    r = rand(Uniform(dot(b,l), dot(b,u)))
    return CQKProblem(d, a, b, r, l, u)
end

function CQK_corr(n)
    b = rand(Uniform(10,25), n)
    d = similar(b)
    a = similar(b)
    @. d = a = b + 5
    r1 = rand(Uniform(10,25), n)
    r2 = rand(Uniform(10,25), n)
    l = min.(r1, r2)
    u = max.(r1, r2)
    r = rand(Uniform(dot(b,l), dot(b,u)))
    return CQKProblem(d, a, b, r, l, u)
end

function cqk_infeas(P, sol)
    return abs(dot(P.b, sol) - P.r) / P.r
end

#################################
# Instances
#################################
CQK_INSTANCES = INSTANCE[]

CQK_names = ["uncorr", "weakly corr", "corr"]
CQK_sizes = [100_000_000, 1_000_000_000] #[1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
CQK_gen   = [n -> CQK_uncorr(n), n -> CQK_wcorr(n), n -> CQK_corr(n)]

unitsize64 = sizeof(CQK_uncorr(1))
maxn = maxsize(unitsize64, USECUDA)
for i in eachindex(CQK_names), n in CQK_sizes
    if n > maxn
        @info "Problem size $n may not fit in RAM or GPU memory, skipping."
        continue
    end
    push!(CQK_INSTANCES, INSTANCE(CQK_names[i], n, CQK_gen[i]))
end

##############
# Methods
##############
CQK_METHODS = METHOD[]

push!(CQK_METHODS, METHOD(
    "cqk (F64)",
    identity,
    (P, nthreads) -> b_dense(P, cqk!, nthreads),
    (P, nthreads) -> cqk(P, nchunks=nthreads)[2:3],
    (P, nthreads) -> cqk_infeas(P, cqk(P, nchunks=nthreads)[1]),
    (P, nthreads) -> Inf
    )
)

push!(CQK_METHODS, METHOD(
    "cqn (F64)",
    identity,
    (P, nthreads) -> b_cms_cqn(P, nthreads),
    (P, nthreads) -> cms_cqn(P)[2:3],
    (P, nthreads) -> cqk_infeas(P, cms_cqn(P)[1]),
    (P, nthreads) -> reldiff_sol(P, cqk, cms_cqn)
    )
)

if USECUDA > 32
    push!(CQK_METHODS, METHOD(
        "cqk (GPU, FP64)",
        P -> CPUtoGPU(P, Float64),
        (P, nthreads) -> b_cuda(P, cqk!, nthreads),
        (P, nthreads) -> cqk(P)[2:3],
        (P, nthreads) -> cqk_infeas(P, cqk(P)[1]),
        (P, nthreads) -> reldiff_sol(P, cqk)
        )
    )
end

push!(CQK_METHODS, METHOD(
    "cqk (CPU, F32)",
    F64toF32,
    (P, nthreads) -> b_dense(P, cqk!, nthreads),
    (P, nthreads) -> cqk(P, nchunks=nthreads)[2:3],
    (P, nthreads) -> cqk_infeas(P, cqk(P, nchunks=nthreads)[1]),
    (P, nthreads) -> reldiff_sol(P, cqk)
    )
)
if USECUDA > 0 
    push!(CQK_METHODS, METHOD(
        "cqk (GPU, FP32)",
        P -> CPUtoGPU(P, Float32),
        (P, nthreads) -> b_cuda(P, cqk!, nthreads),
        (P, nthreads) -> cqk(P)[2:3],
        (P, nthreads) -> cqk_infeas(P, cqk(P)[1]),
        (P, nthreads) -> reldiff_sol(P, cqk)
        )
    )
end
