#real_data_kdd12.jl

using SparseArrays
using BenchmarkTools, Random, Distributions
using StatsBase
using JLD2
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20

include("simplex_and_l1ball/l1ball_wrap.jl")

println("JDL cannot read file, please run following command in the script.")
#=
run following command if JDL cannot load file
include("real_data_reader.jl")
=#
println("You are using ", nthreads(), " threads for parallel computing")
println("Warning: following experiments are for 40 threads!")

I, J, V, label = JLD2.load_object("kdd12_part.jld2")
kdd12 = sparse(I, J, V)
kdd12 = convert(SparseMatrixCSC{Float64,Int64}, kdd12)
label = convert(Array{Float64,1}, label)
label2 = label .- 0.5

function psgd_serial(f::Function, A, label, iter, stepsize, b, toler)
    #get an initial x
    x = sprand(size(A)[1], 0.5)
    y = spzeros(size(A)[1])
    t = 0
    for i in 1:iter
        I = sample(1:size(A)[2], 128; replace=false)
        d = x - 2 * stepsize * A[:, I] * (A[:, I]' * x - label[I])
        if length(findnz(d)[1]) < 10^5
            x_new = f(Array(d), b)
        else
            d = Array(d)
            t1 = time_ns()
            x_new = f(d, b)
            t2 = time_ns()
            t = t + (t2 - t1) / 1.0e9
        end
        y += x_new
        if i % 10 == 0
            x = y ./ 10
            y = spzeros(size(A)[1])
        end
    end
    return t
end

function psgd_parallel(f::Function, A, label, iter, stepsize, b, toler, numthread, spa)
    #get an initial x
    x = sprand(size(A)[1], 0.5)
    y = spzeros(size(A)[1])
    t = 0
    for i in 1:iter
        I = sample(1:size(A)[2], 128; replace=false)
        d = x - 2 * stepsize * A[:, I] * (A[:, I]' * x - label[I])
        if length(findnz(d)[1]) < 10^5
            x_new = f(Array(d), b, numthread, spa)
        else
            d = Array(d)
            t1 = time_ns()
            x_new = f(d, b, numthread, spa)
            t2 = time_ns()
            t = t + (t2 - t1) / 1.0e9
        end
        y += x_new
        if i % 10 == 0
            x = y ./ 10
            y = spzeros(size(A)[1])
        end
    end
    return t
end

function psgd_parallel_sortscan(f::Function, A, label, iter, stepsize, b, toler, numthread)
    #get an initial x
    x = sprand(size(A)[1], 0.5)
    y = spzeros(size(A)[1])
    t = 0
    for i in 1:iter
        I = sample(1:size(A)[2], 128; replace=false)
        d = x - 2 * stepsize * A[:, I] * (A[:, I]' * x - label[I])
        if length(findnz(d)[1]) < 10^5
            x_new = f(Array(d), b, numthread)
        else
            d = Array(d)
            t1 = time_ns()
            x_new = f(d, b, numthread)
            t2 = time_ns()
            t = t + (t2 - t1) / 1.0e9
        end
        y += x_new
        if i % 10 == 0
            x = y ./ 10
            y = spzeros(size(A)[1])
        end
    end
    return t
end

Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_serial(l1ball_sortscan_s, kdd12, label2, 10, 0.01, 1, 0.01)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 8)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 8)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 16)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 16)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 24)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 24)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 32)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 32)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 40)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 40)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 48)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 48)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 56)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 56)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 64)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 64)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 72)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 72)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 80)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel_sortscan(l1ball_sortPscan_p, kdd12, label2, 10, 0.01, 1, 0.01, 80)
end
println(median(t))

Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_serial(l1ball_michelot_s, kdd12, label2, 10, 0.01, 1, 0.01)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_michelot_p, kdd12, label2, 10, 0.01, 1, 0.01, 8, 0.0)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_michelot_p, kdd12, label2, 10, 0.01, 1, 0.01, 16, 0.0)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_michelot_p, kdd12, label2, 10, 0.01, 1, 0.01, 24, 0.0)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_michelot_p, kdd12, label2, 10, 0.01, 1, 0.01, 32, 0.0)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_michelot_p, kdd12, label2, 10, 0.01, 1, 0.01, 40, 0.0)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_michelot_p, kdd12, label2, 10, 0.01, 1, 0.01, 48, 0.0)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_michelot_p, kdd12, label2, 10, 0.01, 1, 0.01, 56, 0.0)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_michelot_p, kdd12, label2, 10, 0.01, 1, 0.01, 64, 0.0)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 72, 0.0)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_michelot_p, kdd12, label2, 10, 0.01, 1, 0.01, 80, 0.0)
end
println(median(t))

Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_serial(l1ball_condat_s, kdd12, label2, 10, 0.01, 1, 0.01)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 8, 0.001)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 16, 0.001)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 24, 0.001)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 32, 0.001)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 40, 0.001)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 48, 0.001)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 56, 0.001)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 64, 0.001)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 72, 0.001)
end
println(median(t))
Random.seed!(12345);
t = zeros(20)
for i in 1:20
    t[i] = psgd_parallel(l1ball_condat_p, kdd12, label2, 10, 0.01, 1, 0.01, 80, 0.001)
end
println(median(t))
