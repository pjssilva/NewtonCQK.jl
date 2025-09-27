#sort_scan.jl

using Base.Threads #for @parallel macro
using BangBang #for append!!() function
using SparseArrays #for sparse vector
using ThreadsX #for parallel sort

"""
    sortscan_pivot(data, b)

For input vector d (or called as data here), sortscan_pivot return τ for
projection onto simplex with scaling factor b based on sort and scan method.
"""
function sortscan_pivot(data::Array{Float64,1}, b::Real=1)::Float64
    #using quicksort here
    d_after_sort = sort(data; alg=QuickSort, rev=true)
    #initialize the summation before scanning
    summation = -b
    for (index, x) in enumerate(d_after_sort)
        summation += x
        #find the τ
        (summation / index > x) && return (summation - x) / (index - 1)
    end
    #here is to prevent function returns nothing when cannot find index
    #such that summation/index > x
    return summation / length(d_after_sort)
end

"""
    sortscan_parallel_pivot(data, b)

The function is similar to sortscan_pivot(), but uses parallel sort and
    parallel prefix sum to return τ
"""
function sortscan_parallel_pivot(data::Array{Float64,1}, b::Real=1)::Float64
    #using parallel mergesort here (not parallel quicksort)
    d_after_sort = sort(data; alg=ThreadsX.MergeSort, rev=true)
    #copy d_after_sort as prefix sum list
    prefix_sum = copy(d_after_sort)
    #do parallel scan
    parallel_prefixsum!(prefix_sum)
    #find τ
    for i in 1:length(prefix_sum)
        ((prefix_sum[i] - b) / i > d_after_sort[i]) &&
            return (pivot = (prefix_sum[i - 1] - b) / (i - 1))
    end
    #here is to prevent function returns nothing when cannot find index
    #such that summation/index > x
    return pivot = (prefix_sum[end] - b) / length(prefix_sum)
end

"""
    parallel_prefixsum!(d)

Parallel prefix sum (or parallel scan) with "butterfly" structure.
"""
function parallel_prefixsum!(d::Array{Float64,1})::Array{Float64,1}
    #record length
    lenght_cache = length(d)
    #how many layers in prefix sum
    k = ceil(Int, log2(lenght_cache))
    #broadcast stage
    for j in 1:k
        @threads for i in (2 ^ j):(2 ^ j):min(lenght_cache, 2 ^ k)
            @inbounds d[i] = d[i - 2 ^ (j - 1)] + d[i]
        end
    end
    #reduce stage
    for j in (k - 1):-1:1
        @threads for i in (3 * 2 ^ (j - 1)):(2 ^ j):min(lenght_cache, 2 ^ k)
            @inbounds d[i] = d[i - 2 ^ (j - 1)] + d[i]
        end
    end
    #return prefix sum result
    return d
end

"""
    parallel_partial_scan(d, b)

Parallel prefix sum with early termination.
"""
function parallel_partial_scan(d::Array{Float64,1}, b::Real=1)::Float64
    #record length
    length_cache = length(d)
    #if the length of input is too small, please use serial algorithm
    if length_cache < 100
        println("Error: the length of y should be at least 100!")
        return nothing
    end
    #calculate the layers of prefix sum
    k = ceil(Int, log2(length_cache))
    #avoid change raw data
    prefix_sum = copy(d)
    #initialize
    stop_layer = copy(k)
    #broadcast stage
    for j in 1:k
        @threads for i in (2 ^ j):(2 ^ j):min(length_cache, 2 ^ k)
            @inbounds prefix_sum[i] += prefix_sum[i - 2 ^ (j - 1)]
        end
        pivot = (prefix_sum[min(length_cache, 2^j)] - b) / min(length_cache, 2^j)
        if pivot >= d[min(length_cache, 2^j)]
            stop_layer = j
            break
        end
    end

    if stop_layer == k
        pivot_full = (sum(d) - b) / length_cache
        (pivot_full < d[length_cache]) && return pivot_full
    end

    #reduce stage
    st = 2^(stop_layer - 1)
    for j in (stop_layer - 1):-1:1
        i = min(st + 2^(j - 1), length_cache)
        prefix_sum[i] += prefix_sum[st]
        pivot = (prefix_sum[i] - b) / i
        if pivot < d[i]
            st = i
        elseif pivot == d[i] || i == length_cache
            break
        end
    end

    return (prefix_sum[st] - b) / st
end

"""
    scan(data, pivot)
Returns max.(data.-pivot, 0) with sparse structure
"""
function scan(data::Array{Float64,1}, pivot::Real)::AbstractVector
    #initialize two sets
    I_p = [] #active index set
    V = [] #value set
    for i in 1:length(data)
        if data[i] > pivot
            push!(I_p, i)
            push!(V, data[i] - pivot)
        end
    end
    #argument "length(data)" guarantees the length of sparse vector
    #is identical to the input data
    return sparsevec(I_p, V, length(data))
end

"""
    pscan(data, pivot, numthread)

Similar to scan(), but using multiple threads in max.(data[i]-pivot, 0).
"""
function pscan(data::Array{Float64,1}, pivot::Real, numthread::Int=1)::AbstractVector
    I_p = []
    V = []
    width = ceil(Int, length(data) / numthread)
    spl = SpinLock()
    @threads for i in 1:width:length(data)
        let
            st = i
            en = min(i + width - 1, length(data))
            local_I = []
            local_V = []
            for j in st:en
                local x = data[j] - pivot
                if x > 0
                    push!(local_I, j)
                    push!(local_V, x)
                end
            end
            lock(spl)
            append!!(I_p, local_I)
            append!!(V, local_V)
            unlock(spl)
        end
    end
    push!(I_p, length(data) + 1)
    push!(V, 0)
    return sparsevec(I_p, V)[1:length(data)]
end
