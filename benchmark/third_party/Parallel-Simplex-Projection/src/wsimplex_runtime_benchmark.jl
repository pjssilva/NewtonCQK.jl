#wsimplex_runtime_benchmark.jl

using BenchmarkTools, Random, Distributions
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20

include("weighted_simplex_and_ball/wsortscan.jl")
include("weighted_simplex_and_ball/wmichelot.jl")
include("weighted_simplex_and_ball/wcondat.jl")

println("You are using ", nthreads(), " threads for parallel computing")
println("Warning: following experiments are for 80 threads!")

function wsortscan_test()
    Random.seed!(12345)
    res = @benchmark wsortscan_s($(rand(Normal(0, 1), 10^8)), $(rand(10^8)), 1)
    println("serial: ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark wsortscan_p($(rand(Normal(0, 1), 10^8)), $(rand(10^8)), 1)
        println(i * 8, " : ", median(res))
    end
end

function wmichelot_test()
    Random.seed!(12345)
    res = @benchmark wmichelot_s($(rand(Normal(0, 1), 10^8)), $(rand(10^8)), 1)
    println("serial: ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark wmichelot_p(
            $(rand(Normal(0, 1), 10^8)), $(rand(10^8)), 1, $(i * 8)
        )
        println(i * 8, " : ", median(res))
    end
end

function wcondat_test()
    Random.seed!(12345)
    res = @benchmark wcondat_s($(rand(Normal(0, 1), 10^8)), $(rand(10^8)), 1)
    println("serial: ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark wcondat_p($(rand(Normal(0, 1), 10^8)), $(rand(10^8)), 1, $(i * 8))
        println(i * 8, " : ", median(res))
    end
end

wsortscan_test()
wmichelot_test()
wcondat_test()
