using NewtonCQK
using Test
using Printf
using ProximalOperators
using Random: Random
using LinearAlgebra

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
        soln, iter, flag = simplex_proj(x, r=rhs)
        @test flag == :solved
        @test soln ≈ solp

        # Preallocate workspace
        workspc = initialize_chunks(n)
        soln, iter, flag = simplex_proj(x; r=rhs, chunks=workspc)
        @test flag == :solved
        @test soln ≈ solp

        # Preallocate output
        soln = similar(x)
        iter, flag = simplex_proj!(soln, x; r=rhs)
        @test flag == :solved
        @test soln ≈ solp

        # Sparse vectors
        soln, iter, flag = spsimplex_proj(x; r=rhs, chunks=workspc)
        @test flag == :solved
        @test Vector(soln) ≈ solp
    end
end

@testset "L1 Projection" begin

    # Verify if first coordinate is used when estimating 1-norm
    x =[8.750000000000004, 1.249999999999999, 1.2499999999999998, -1.2500000000000018, 1.2500000000000007, -1.2500000000000007, -1.2500000000000018, 1.2499999999999996] 
    r = 9.354143466934856 
    sol, iter, flag = l1ball_proj(x, r=r)
    @test norm(sol, 1) ≈ r

end

# TODO: There are a lot of tests to make. L1-Ball and general CQK, variations
# of paramters.
