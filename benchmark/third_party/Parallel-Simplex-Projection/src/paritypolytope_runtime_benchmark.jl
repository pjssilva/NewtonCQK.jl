#paritypolytope_runtime_benchmark.jl

using BenchmarkTools, Random, Distributions
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20

println("You are using ", nthreads(), " threads for parallel computing")

include("parity_polytope.jl")
Random.seed!(12345);
res1 = @benchmark parity_s(sortscan_s, $(rand(10^8 - 1) .+ 1));
println(res1);
Random.seed!(12345);
res2 = @benchmark sortscan_s($(parity_polytope_prepare(rand(10^8 - 1) .+ 1)), 1);
println(res2);
Random.seed!(12345);
res1 = @benchmark parity_p_sortscan(sortscan_p, $(rand(10^8 - 1) .+ 1));
println(res1);
Random.seed!(12345);
res2 = @benchmark sortscan_p($(parity_polytope_prepare(rand(10^8 - 1) .+ 1)), 1, nthreads());
println(res2);
Random.seed!(12345);
res1 = @benchmark parity_p_sortscan(sortPscan_p, $(rand(10^8 - 1) .+ 1));
println(res1);
Random.seed!(12345);
res2 = @benchmark sortPscan_p(
    $(parity_polytope_prepare(rand(10^8 - 1) .+ 1)), 1, nthreads()
);
println(res2);

Random.seed!(12345);
res1 = @benchmark parity_s(michelot_s, $(rand(10^8 - 1) .+ 1));
println(res1);
Random.seed!(12345);
res2 = @benchmark michelot_s($(parity_polytope_prepare(rand(10^8 - 1) .+ 1)), 1);
println(res2);
Random.seed!(12345);
res1 = @benchmark parity_p(michelot_p, $(rand(10^8 - 1) .+ 1), 0.0);
println(res1);
Random.seed!(12345);
res2 = @benchmark michelot_p(
    $(parity_polytope_prepare(rand(10^8 - 1) .+ 1)), 1, nthreads(), 0.0
);
println(res2);

Random.seed!(12345);
res1 = @benchmark parity_s(condat_s, $(rand(10^8 - 1) .+ 1));
println(res1);
Random.seed!(12345);
res2 = @benchmark condat_s($(parity_polytope_prepare(rand(10^8 - 1) .+ 1)), 1);
println(res2);
Random.seed!(12345);
res1 = @benchmark parity_p(condat_p, $(rand(10^8 - 1) .+ 1), 0.001);
println(res1);
Random.seed!(12345);
res2 = @benchmark condat_p(
    $(parity_polytope_prepare(rand(10^8 - 1) .+ 1)), 1, nthreads(), 0.001
);
println(res2);
