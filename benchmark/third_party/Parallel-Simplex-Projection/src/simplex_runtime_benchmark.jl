#simplex_runtime_benchmark

using BenchmarkTools, Random, Distributions
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20

include("simplex_and_l1ball/simplex_wrap.jl")

println("You are using ", nthreads(), " threads for parallel computing")
println("Warning: following experiments are for 80 threads!")

#=
Following code is for our paper experiments, 80 threads and testing results for
1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80 threads
=#
function standard_normal_test_condat(n::Int)
    println("results for condat:")
    Random.seed!(12345)
    res = @benchmark condat_s($(rand(Normal(0, 1), n)), 1)
    println("serial : ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark condat_p($(rand(Normal(0, 1), n)), 1, $(i * 8), 0.01)
        println(i * 8, " : ", median(res))
    end
end

function uniform_test_condat(n::Int)
    println("results for condat:")
    Random.seed!(12345)
    res = @benchmark condat_s($(rand(n)), 1)
    println("serial : ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark condat_p($(rand(n)), 1, $(i * 8), 0.00001)
        println(i * 8, " : ", median(res))
    end
end

function small_variance_normal_test_condat(n::Int)
    println("results for condat:")
    Random.seed!(12345)
    res = @benchmark condat_s($(rand(Normal(0, 0.001), n)), 1)
    println("serial : ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark condat_p($(rand(Normal(0, 0.001), n)), 1, $(i * 8), 0.00001)
        println(i * 8, " : ", median(res))
    end
end

function standard_normal_test_michelot(n::Int)
    println("results for michelot:")
    Random.seed!(12345)
    res = @benchmark michelot_s($(rand(Normal(0, 1), n)), 1)
    println("serial : ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark michelot_p($(rand(Normal(0, 1), n)), 1, $(i * 8), 0.0)
        println(i * 8, " : ", median(res))
    end
end

function uniform_test_michelot(n::Int)
    println("results for michelot:")
    Random.seed!(12345)
    res = @benchmark michelot_s($(rand(n)), 1)
    println("serial : ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark michelot_p($(rand(n)), 1, $(i * 8), 0.0)
        println(i * 8, " : ", median(res))
    end
end

function small_variance_normal_test_michelot(n::Int)
    println("results for michelot:")
    Random.seed!(12345)
    res = @benchmark michelot_s($(rand(Normal(0, 0.001), n)), 1)
    println("serial : ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark michelot_p($(rand(Normal(0, 0.001), n)), 1, $(i * 8), 0.00001)
        println(i * 8, " : ", median(res))
    end
end

function standard_normal_test_sortscan(n::Int)
    println("results for sortscan:")
    Random.seed!(12345)
    res = @benchmark sortscan_s($(rand(Normal(0, 1), n)), 1)
    println("serial : ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark sortscan_p($(rand(Normal(0, 1), n)), 1, $(i * 8))
        println(i * 8, " sortscan : ", median(res))
        Random.seed!(12345)
        res = @benchmark sortPscan_p($(rand(Normal(0, 1), n)), 1, $(i * 8))
        println(i * 8, " sortPscan: ", median(res))
    end
end

function uniform_test_sortscan(n::Int)
    println("results for sortscan:")
    Random.seed!(12345)
    res = @benchmark sortscan_s($(rand(n)), 1)
    println("serial : ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark sortscan_p($(rand(n)), 1, $(i * 8))
        println(i * 8, " sortscan : ", median(res))
        Random.seed!(12345)
        res = @benchmark sortPscan_p($(rand(n)), 1, $(i * 8))
        println(i * 8, " sortPscan: ", median(res))
    end
end

function small_variance_normal_test_sortscan(n::Int)
    println("results for sortscan:")
    Random.seed!(12345)
    res = @benchmark sortscan_s($(rand(Normal(0, 0.001), n)), 1)
    println("serial : ", median(res))
    for i in 1:10
        Random.seed!(12345)
        res = @benchmark sortscan_p($(rand(Normal(0, 0.001), n)), 1, $(i * 8))
        println(i * 8, " sortscan : ", median(res))
        Random.seed!(12345)
        res = @benchmark sortPscan_p($(rand(Normal(0, 0.001), n)), 1, $(i * 8))
        println(i * 8, " sortPscan: ", median(res))
    end
end

# for n = 10^8
uniform_test_sortscan(10^8)
standard_normal_test_sortscan(10^8)
small_variance_normal_test_sortscan(10^8)
uniform_test_michelot(10^8)
standard_normal_test_michelot(10^8)
small_variance_normal_test_michelot(10^8)
uniform_test_condat(10^8)
standard_normal_test_condat(10^8)
small_variance_normal_test_condat(10^8)

# for n = 10^7, 10^8, 10^9
standard_normal_test_sortscan(10^7)
standard_normal_test_michelot(10^7)
standard_normal_test_condat(10^7)
standard_normal_test_sortscan(10^8)
standard_normal_test_michelot(10^8)
standard_normal_test_condat(10^8)
standard_normal_test_sortscan(10^9)
standard_normal_test_michelot(10^9)
standard_normal_test_condat(10^9)
