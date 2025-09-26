#fairness_test.jl

using BenchmarkTools, Random, Distributions
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20

include("simplex_and_l1ball/simplex_wrap.jl")

println("You are using ", nthreads(), " threads for parallel computing")
if nthreads() != 1
    println("Warning: following experiments are for 1 threads!")
end

Random.seed!(12345);
res1 = @benchmark sortscan_s($(rand(100_000_000)), 1);
Random.seed!(12345);
res2 = @benchmark sortscan_p($(rand(100_000_000)), 1, 1);
Random.seed!(12345);
res3 = @benchmark sortPscan_p($(rand(100_000_000)), 1, 1);
println("serial: ", median(res1))
println("parallel sortscan: ", median(res2))
println("parallel sortPscan: ", median(res3))

Random.seed!(12345);
res1 = @benchmark michelot_s($(rand(100_000_000)), 1);
Random.seed!(12345);
res2 = @benchmark michelot_p($(rand(100_000_000)), 1, 1, 0.0001);
println("serial: ", median(res1))
println("parallel: ", median(res2))

Random.seed!(12345);
res1 = @benchmark condat_s($(rand(100_000_000)), 1);
Random.seed!(12345);
res2 = @benchmark condat_p($(rand(100_000_000)), 1, 1, 0.01);
println("serial: ", median(res1))
println("parallel: ", median(res2))
