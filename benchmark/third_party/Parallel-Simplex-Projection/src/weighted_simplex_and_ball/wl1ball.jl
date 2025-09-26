#wl1ball.jl

using Base.Threads
using BangBang
using SparseArrays
using ThreadsX

include("wsortscan.jl")
include("wmichelot.jl")
include("wcondat.jl")

"""
    wball_sortscan_s(data, w, b)

Weighted l1 ball projection based on serial sort and scan method
"""
function wball_sortscan_s(
    data::Array{Float64,1}, w::Array{Float64,1}, b::Real
)::AbstractVector
    let
        d = abs.(data)
        #define a new array to store data/w, data, w
        y = []
        for i in 1:length(data)
            push!(y, [d[i] / w[i], d[i], w[i]])
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
                if data[i] >= 0
                    x[i] = d[i] - w[i] * pivot
                else
                    x[i] = -d[i] + w[i] * pivot
                end
            end
        end
        #output final result
        return x
    end
end

"""
    parallel_prefixsum!(y)

Parallel prefix sum (parallel scan) algorithm
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
    wball_sortscan_p(data, w, b)

Weighted l1 ball projection based on parallel sort and scan method
"""
function wball_sortscan_p(
    data::Array{Float64,1}, w::Array{Float64,1}, b::Real
)::AbstractVector
    let
        d = abs.(data)
        #define a new array to store data/w, data, w
        y = []
        for i in 1:length(data)
            push!(y, [d[i] / w[i], d[i], w[i]])
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
        pivot = 0
        for i in 1:length(data)
            if ((x1[i] - b) / x2[i]) >= y[i][1]
                p = (x1[i - 1] - b) / x2[i - 1]
                break
            end
        end
        #initialize final list
        x = zeros(length(data))
        @threads for i in 1:length(data)
            if data[i] > w[i] * pivot
                if data[i] >= 0
                    @inbounds x[i] = d[i] - w[i] * pivot
                else
                    @inbounds x[i] = -d[i] + w[i] * pivot
                end
            end
        end
        #output final result
        return x
    end
end

"""
    wball_michelot_s(data, w, b)

Weighted l1 ball projection based on serial Michelot's method
"""
function wball_michelot_s(
    data::Array{Float64,1}, w::Array{Float64,1}, b::Real
)::AbstractVector
    let
        d = abs.(data)
        s1 = d' * w
        s2 = w' * w
        pivot = (s1 - b) / s2
        active_list = [i for i in 1:length(d)]
        while true
            length_cache = length(active_list)
            for _ in 1:length_cache
                i = popfirst!(active_list)
                if d[i] / w[i] > pivot
                    push!(active_list, i)
                else
                    s1 = s1 - d[i] * w[i]
                    s2 = s2 - w[i]^2
                end
            end
            if length_cache == length(active_list)
                break
            end
            pivot = (s1 - b) / s2
        end

        value_list = Float64[]
        for j in active_list
            if data[j] >= 0
                push!(value_list, d[j] - w[j] * pivot)
            else
                push!(value_list, -d[j] + w[j] * pivot)
            end
        end
        return sparsevec(active_list, value_list, length(data))
    end
end

"""
    wball_michelot_p(data, w, b, numthread)

Weighted l1 ball projection based on parallel Michelot's method
"""
function wball_michelot_p(
    data::Array{Float64,1}, w::Array{Float64,1}, b::Real, numthread::Int
)
    let
        #the length for subvectors
        width = floor(Int, length(data) / numthread)
        #lock global value
        spl = SpinLock()
        #initialize a global list
        glist = Int[]
        gs1 = 0.0
        gs2 = 0.0
        @threads for id in 1:numthread
            let
                #determine start and end position for subvectors
                local st = (id - 1) * width + 1
                if id == numthread
                    local en = length(data)
                else
                    local en = id * width
                end
                local d = abs.(data[st:en])
                local active_list = [i for i in st:en]
                local s1 = d' * w[st:en]
                local s2 = w[st:en]' * w[st:en]
                local pivot = (s1 - b) / (s2)
                while true
                    length_cache = length(active_list)
                    for i in 1:length_cache
                        j = popfirst!(active_list)
                        if d[i] / w[j] > pivot
                            push!(active_list, i)
                        else
                            s1 = s1 - d[i] * w[j]
                            s2 = s2 - w[j]^2
                        end
                    end
                    if length_cache == length(active_list)
                        break
                    end
                    pivot = (s1 - b) / s2
                end
                #reduce with locking
                lock(spl)
                append!!(glist, active_list)
                gs1 += s1
                gs2 += s2
                unlock(spl)
            end
        end
        pivot = (gs1 - b) / gs2
        while true
            length_cache = length(glist)
            for _ in 1:length_cache
                i = popfirst!(glist)
                if abs(data[i]) / w[i] > pivot
                    push!(glist, i)
                else
                    gs1 = gs1 - abs(data[i]) * w[i]
                    gs2 = gs2 - w[i]^2
                end
            end
            if length_cache == length(glist)
                break
            end
            pivot = (gs1 - b) / gs2
        end

        value_list = Float64[]
        for j in glist
            if data[j] >= 0
                push!(value_list, abs(data[j]) - w[j] * pivot)
            else
                push!(value_list, -abs(data[j]) + w[j] * pivot)
            end
        end
        return sparsevec(glist, value_list, length(data))
    end
end

"""
    wcheckL_ball(active_list, s1, s2, d, data, w, b)

Remaining part (after Filter) of Condat's algorithm in weighted l1 ball projection
"""
function wcheckL_ball(
    active_list::Array{Int,1},
    s1::Float64,
    s2::Float64,
    d::Array{Float64,1},
    data::Array{Float64,1},
    w::Array{Float64,1},
    b::Real,
)::AbstractVector
    let
        pivot = (s1 - b) / s2
        while true
            length_cache = length(active_list)
            for _ in 1:length_cache
                i = popfirst!(active_list)
                if d[i] / w[i] > pivot
                    push!(active_list, i)
                else
                    s1 = s1 - d[i] * w[i]
                    s2 = s2 - w[i]^2
                    pivot = (s1 - b) / s2
                end
            end
            if length_cache == length(active_list)
                break
            end
        end

        value_list = Float64[]
        for j in active_list
            if data[j] >= 0
                push!(value_list, d[j] - w[j] * pivot)
            else
                push!(value_list, -d[j] + w[j] * pivot)
            end
        end
        return sparsevec(active_list, value_list, length(data))
    end
end

"""
    parallel_wfilter(data, w, b, numthread)

Parallel filter for projection onto weighted l1 ball
"""
function parallel_wfilter(
    data::Array{Float64,1}, w::Array{Float64,1}, b::Real, numthread::Int
)
    #the length for subvectors
    width = floor(Int, length(data) / numthread)
    #lock global value
    spl = SpinLock()
    #initialize a global list
    glist = Int[]
    gs1 = 0.0
    gs2 = 0.0
    @threads for id in 1:numthread
        let
            #determine start and end position for subvectors
            local st = (id - 1) * width + 1
            if id == numthread
                local en = length(data)
            else
                local en = id * width
            end
            local d = abs.(data[st:en])
            local active_list = Int[st]
            local s1 = d[1] * w[st]
            local s2 = w[1]^2
            local pivot = (s1 - b) / (s2)
            local wait_list = Int[]
            #check all terms
            for i in (st + 1):en
                #remove inactive terms
                if d[i - st + 1] / w[i] > pivot
                    #update pivot
                    pivot = (s1 + d[i - st + 1] * w[i] - b) / (s2 + w[i]^2)
                    if pivot > (d[i - st + 1] * w[i] - b) / (w[i]^2)
                        push!(active_list, i)
                        s1 += d[i - st + 1] * w[i]
                        s2 += w[i]^2
                    else
                        #for large pivot
                        append!!(wait_list, active_list)
                        active_list = Int[i]
                        s1 = d[i - st + 1] * w[i]
                        s2 = w[i]^2
                        pivot = (s1 - b) / s2
                    end
                end
            end
            #reuse terms from waiting list
            for j in wait_list
                if d[j - st + 1] / w[j] > pivot
                    push!(active_list, j)
                    s1 += d[j - st + 1] * w[j]
                    s2 += w[j]^2
                    pivot = (s1 - b) / s2
                end
            end
            while true
                length_cache = length(active_list)
                for _ in 1:length_cache
                    i = popfirst!(active_list)
                    if d[i - st + 1] / w[i] > pivot
                        push!(active_list, i)
                    else
                        s1 = s1 - d[i - st + 1] * w[i]
                        s2 = s2 - w[i]^2
                        pivot = (s1 - b) / s2
                    end
                end
                if length_cache == length(active_list)
                    break
                end
            end
            #reduce with locking
            lock(spl)
            append!!(glist, active_list)
            gs1 += s1
            gs2 += s2
            unlock(spl)
        end
    end
    return glist, gs1, gs2
end

"""
    wcheckL_ball_p(active_list, s1, s2, data, w, b)

Remaining part (after Filter) of Condat's algorithm in weighted l1 ball projection
"""
function wcheckL_ball_p(
    active_list::Array{Int,1},
    s1::Float64,
    s2::Float64,
    data::Array{Float64,1},
    w::Array{Float64,1},
    b::Real,
)::AbstractVector
    let
        pivot = (s1 - b) / s2
        while true
            length_cache = length(active_list)
            for _ in 1:length_cache
                i = popfirst!(active_list)
                x = abs.(data[i])
                if x / w[i] > pivot
                    push!(active_list, i)
                else
                    s1 = s1 - x * w[i]
                    s2 = s2 - w[i]^2
                    pivot = (s1 - b) / s2
                end
            end
            if length_cache == length(active_list)
                break
            end
        end

        value_list = Float64[]
        for j in active_list
            if data[j] >= 0
                push!(value_list, abs.(data[j]) - w[j] * pivot)
            else
                push!(value_list, -abs.(data[j]) + w[j] * pivot)
            end
        end
        return sparsevec(active_list, value_list, length(data))
    end
end

"""
    wball_condat_s(data, w, b)

Weighted l1 ball projection based on serial Condat's method
"""
function wball_condat_s(
    data::Array{Float64,1}, w::Array{Float64,1}, b::Real
)::AbstractVector
    d = abs.(data)
    active_list, s1, s2 = wfilter(d, w, b)
    return wcheckL_ball(active_list, s1, s2, d, data, w, b)
end

"""
    wball_condat_p(data, w, b, numthread)

Weighted l1 ball projection based on parallel Condat's method
"""
function wball_condat_p(
    data::Array{Float64,1}, w::Array{Float64,1}, b::Real, numthread
)::AbstractVector
    active_list, s1, s2 = parallel_wfilter(data, w, b, numthread)
    return wcheckL_ball_p(active_list, s1, s2, data, w, b)
end
