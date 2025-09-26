#wsortscan.jl

using Base.Threads
using BangBang
using ThreadsX

"""
    wsortscan_s(data, w, b)

Weighted simplex projetion based on serial sort and scan method
"""
function wsortscan_s(data::Array{Float64,1}, w::Array{Float64,1}, b::Real)::AbstractVector
    let
        #define a new array to store data/w, data, w
        y = []
        for i in 1:length(data)
            push!(y, [data[i] / w[i], data[i], w[i]])
        end
        #sort y based on data/w
        y = sort(y; alg=QuickSort, rev=true)
        #initialize values
        s1 = 0
        s2 = 0
        pivot = 0
        #check all terms
        for i in 1:length(data)
            s1 += y[i][2] * y[i][3]
            s2 += y[i][3]^2
            if (s1 - b) / s2 >= y[i][2] / y[i][3]
                pivot = (s1 - y[i][2] * y[i][3] - b) / (s2 - y[i][3]^2)
                break
            end
        end
        #initialize final list
        x = zeros(length(data))
        for i in 1:length(data)
            if data[i] > w[i] * pivot
                x[i] = data[i] - w[i] * pivot
            end
        end
        #output final result
        return x
    end
end

"""
    parallel_prefixsum!(y)

Parallel prefix sum (parallel scan)
"""
function parallel_prefixsum!(y::Array{Float64,1})
    l = length(y)
    k = ceil(Int, log2(l))
    for j in 1:k
        @threads for i in (2^j):(2^j):min(l, 2^k)
            @inbounds y[i] = y[i - 2^(j - 1)] + y[i]
        end
    end
    for j in (k - 1):-1:1
        @threads for i in (3 * 2^(j - 1)):(2^j):min(l, 2^k)
            @inbounds y[i] = y[i - 2^(j - 1)] + y[i]
        end
    end
    return y
end

"""
    wsortscan_p(data, w, b)

Weighted simplex projection based on parallel sort and scan method
"""
function wsortscan_p(data::Array{Float64,1}, w::Array{Float64,1}, b::Real)::AbstractVector
    let
        #define a new array to store data/w, data, w
        y = []
        for i in 1:length(data)
            push!(y, [data[i] / w[i], data[i], w[i]])
        end
        #sort y based on data/w
        y = sort(y; alg=ThreadsX.MergeSort, rev=true)
        #initialize values
        x1 = zeros(length(data))
        x2 = zeros(length(data))
        @threads for i in 1:length(data)
            @inbounds x1[i] = y[i][2] * y[i][3]
            @inbounds x2[i] = y[i][3]^2
        end
        #get prifix sum
        parallel_prefixsum!(x1)
        parallel_prefixsum!(x2)
        #initialize pivot
        p = 0
        for i in 1:length(data)
            if ((x1[i] - b) / x2[i]) >= y[i][1]
                p = (x1[i - 1] - b) / x2[i - 1]
                break
            end
        end
        #initialize final list
        x = zeros(length(data))
        @threads for i in 1:length(data)
            if data[i] > w[i] * p
                @inbounds x[i] = data[i] - w[i] * p
            end
        end
        #output final result
        return x
    end
end
