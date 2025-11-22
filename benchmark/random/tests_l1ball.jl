#################################
# Auxiliary functions
################################

function l1ball_infeas(sol)
    abss = sum(abs.(sol))
    return (abss <= 1.0) ? 0.0 : abs(sum(abss) - 1.0)
end

##############
# Instances
##############

# The same for simplex

##############
# Methods
##############
l1BALL_METHODS = METHOD[]

# Simplex, dense, CPU, Float64
if USECUDA < 32
    push!(
        l1BALL_METHODS,
        METHOD(
            "l1ball (CPU, FP64)",
            identity,
            (P, nthreads) -> b_dense(P, l1ball_proj!, nthreads),
            (P, nthreads) -> l1ball_proj(P; nchunks=nthreads)[2:3],
            (P, nthreads) -> l1ball_infeas(l1ball_proj(P; nchunks=nthreads)[1]),
            (P, nthreads) -> Inf
        )
    )

    # Simplex, sparse, CPU, Float64
    push!(
        l1BALL_METHODS,
        METHOD(
            "sp l1ball (CPU, FP64)",
            identity,
            (P, nthreads) -> b_sparse(P, spl1ball_proj, nthreads),
            (P, nthreads) -> spl1ball_proj(P; nchunks=nthreads)[2:3],
            (P, nthreads) -> l1ball_infeas(spl1ball_proj(P; nchunks=nthreads)[1]),
            (P, nthreads) -> reldiff_sol(P, l1ball_proj)
        )
    )

    # Parallel Condat l1ball, sparse, CPU, Float64
    push!(
        l1BALL_METHODS,
        METHOD(
            "P Condat (l1ball)",
            identity,
            (P, nthreads) -> b_Pcondat(P, l1ball_condat_s, l1ball_condat_p, nthreads),
            (P, nthreads) -> (-1, :solved),
            (P, nthreads) -> if (nthreads == 1)
                l1ball_infeas(l1ball_condat_s(P, 1.0))
            else
                l1ball_infeas(l1ball_condat_p(P, 1.0, nthreads, 0.001))
            end,
            (P, nthreads) -> reldiff_sol(P, l1ball_proj)
        )
    )
end

# Simplex, dense, GPU, Float64
if USECUDA > 32
    push!(
        l1BALL_METHODS,
        METHOD(
            "l1ball (GPU, FP64)",
            P -> CPUtoGPU(P, Float64),
            (P, nthreads) -> b_cuda(P, l1ball_proj!, nthreads),
            (P, nthreads) -> l1ball_proj(P)[2:3],
            (P, nthreads) -> l1ball_infeas(l1ball_proj(P)[1]),
            (P, nthreads) -> reldiff_sol(P, l1ball_proj)
        )
    )
end

if USECUDA < 32
    # Simplex, dense, CPU, Float32
    push!(
        l1BALL_METHODS,
        METHOD(
            "l1ball (CPU, FP32)",
            F64toF32,
            (P, nthreads) -> b_dense(P, l1ball_proj!, nthreads),
            (P, nthreads) -> l1ball_proj(P; nchunks=nthreads)[2:3],
            (P, nthreads) -> l1ball_infeas(l1ball_proj(P; nchunks=nthreads)[1]),
            (P, nthreads) -> reldiff_sol(P, l1ball_proj)
        )
    )
else
    # Simplex, dense, GPU, Float32
    push!(
        l1BALL_METHODS,
        METHOD(
            "l1ball (GPU, FP32)",
            P -> CPUtoGPU(P, Float32),
            (P, nthreads) -> b_cuda(P, l1ball_proj!, nthreads),
            (P, nthreads) -> l1ball_proj(P)[2:3],
            (P, nthreads) -> l1ball_infeas(l1ball_proj(P)[1]),
            (P, nthreads) -> reldiff_sol(P, l1ball_proj)
        )
    )
end
