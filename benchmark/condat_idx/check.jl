# Correctness check: the index-based variant of Condat's method must reproduce
# the output of Condat's original code bit for bit.
#
#   julia --project condat_idx/check.jl
using NewtonCQK, SparseArrays, Random, Printf

include("../third_party/condat/condat_interface.jl")

function rand_nonzerovec(gen, n)
    x = gen(n)
    while any(x .== 0.0)
        x .= gen(n)
    end
    return x
end

gens = [("Random 1", n -> rand(n)), ("Random 2", n -> randn(n)), ("Random 3", n -> 1e-3 .* randn(n))]
bits(v) = reinterpret(UInt64, v)

Random.seed!(2026)
nfail = 0
for (name, gen) in gens, n in [1, 2, 3, 10, 1_000, 10_000, 100_000, 1_000_000], rep in 1:5
    y = rand_nonzerovec(gen, n)
    x0 = condat_proj(y)
    x1 = condat_idx_proj(y)
    sp = condat_idx_spproj(y)
    # In place (x == y)
    z0 = copy(y); condat_proj!(z0, z0)
    z1 = copy(y); condat_idx_proj!(z1, z1)
    xn = simplex_proj(y; nchunks=1)[1]
    ok = bits(x0) == bits(x1) && bits(x0) == bits(Vector(sp)) && bits(z0) == bits(z1)
    ok &= bits(x0) == bits(z0)
    if !ok
        global nfail += 1
        @printf("FAIL %s n=%d rep=%d\n", name, n, rep)
    end
    if rep == 1
        @printf("%-8s n=%8d  nnz=%7d  identical=%s  max|condat-newton|=%.1e\n",
                name, n, nnz(sp), ok, maximum(abs.(x0 .- xn)))
    end
end
# Degenerate cases
for y in ([1.0, 1.0, 1.0], [0.5, 0.5], [2.0, -1.0, 2.0, -1.0], fill(1e-3, 1000))
    ok = bits(condat_proj(y)) == bits(condat_idx_proj(y)) == bits(Vector(condat_idx_spproj(y)))
    ok || (global nfail += 1; println("FAIL degenerate ", y[1:min(end, 4)]))
end
println(nfail == 0 ? "ALL IDENTICAL" : "$nfail FAILURES")
