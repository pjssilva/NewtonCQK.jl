using NewtonCQK
using Test
using Printf
using ProximalOperators
using Random: Random

function random_x(n)
    fillin = max(1, Int(floor(rand()^2 * n)))
    inds = rand(1:n, fillin)
    x = zeros(n)
    x[inds] = rand(fillin)
    return x
end

@testset "Simplex projection" begin
    # Projection onto the unit simplex

    Random.seed!(0)
    for reps in 1:10
        n = rand([1000, 10_000, 100_000, 1_000_000])
        rhs = rand([1.0, 10*rand(), 1.0e-3*rand(), 1.0e+6*rand()])
        x = random_x(n)
        prox_proj(x) = prox(IndSimplex(rhs), x)
        solp, _ = prox_proj(x)

        # Simplest call
        soln, iter = simplex_proj(x, r=rhs)
        @test iter > 0
        @test soln ≈ solp

        # Preallocate workspace
        workspc = initialize_chunks(n)
        soln, iter = simplex_proj(x; r=rhs, chunks=workspc)
        @test iter > 0
        @test soln ≈ solp

        # Preallocate output
        soln = similar(x)
        iter = simplex_proj!(soln, x; r=rhs)
        @test iter > 0
        @test soln ≈ solp

        # Sparse vectors
        soln, iter = spsimplex_proj(x; r=rhs, chunks=workspc)
        @test iter > 0
        @test Vector(soln) ≈ solp
    end
end

# TODO: There are a lot of tests to make. L1-Ball and general CQK, variations
# of paramters.
