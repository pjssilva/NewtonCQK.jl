#wcondat.jl

using Base.Threads
using BangBang
using SparseArrays

"""
    wfilter(data, w, b)

Filter technique for weighted simplex projection
"""
function wfilter(data::Array{Float64,1}, w::Array{Float64,1}, b::Real)
    let
        #initialize
        active_list = Int[1]
        s1 = data[1] * w[1]
        s2 = w[1]^2
        pivot = (s1 - b) / (s2)
        wait_list = Int[]
        #check all terms
        for i in 2:length(data)
            #remove inactive terms
            if data[i] / w[i] > pivot
                #update pivot
                pivot = (s1 + data[i] * w[i] - b) / (s2 + w[i]^2)
                if pivot > (data[i] * w[i] - b) / (w[i]^2)
                    push!(active_list, i)
                    s1 += data[i] * w[i]
                    s2 += w[i]^2
                else
                    #for large pivot
                    append!!(wait_list, active_list)
                    active_list = Int[i]
                    s1 = data[i] * w[i]
                    s2 = w[i]^2
                    pivot = (s1 - b) / s2
                end
            end
        end
        #reuse terms from waiting list
        for j in wait_list
            if data[j] / w[j] > pivot
                push!(active_list, j)
                s1 += data[j] * w[j]
                s2 += w[j]^2
                pivot = (s1 - b) / s2
            end
        end
        return active_list, s1, s2
    end
end

"""
    wcheckL(active_list, s1, s2, data, w, b)

Remaining algorithm (after Filter) of weighted simplex projection based on Condat's method
"""
function wcheckL(
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
                if data[i] / w[i] > pivot
                    push!(active_list, i)
                else
                    s1 = s1 - data[i] * w[i]
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
            push!(value_list, data[j] - w[j] * pivot)
        end
        return sparsevec(active_list, value_list, length(data))
    end
end

"""
    parallel_wfilter(data, w, b, numthread)

Parallel filter technique for weighted simplex projection
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
            local active_list = Int[st]
            local s1 = data[st] * w[st]
            local s2 = w[st]^2
            local pivot = (s1 - b) / (s2)
            local wait_list = Int[]
            #check all terms
            for i in (st + 1):en
                #remove inactive terms
                if data[i] / w[i] > pivot
                    #update pivot
                    pivot = (s1 + data[i] * w[i] - b) / (s2 + w[i]^2)
                    if pivot > (data[i] * w[i] - b) / (w[i]^2)
                        push!(active_list, i)
                        s1 += data[i] * w[i]
                        s2 += w[i]^2
                    else
                        #for large pivot
                        append!!(wait_list, active_list)
                        active_list = Int[i]
                        s1 = data[i] * w[i]
                        s2 = w[i]^2
                        pivot = (s1 - b) / s2
                    end
                end
            end
            #reuse terms from waiting list
            for j in wait_list
                if data[j] / w[j] > pivot
                    push!(active_list, j)
                    s1 += data[j] * w[j]
                    s2 += w[j]^2
                    pivot = (s1 - b) / s2
                end
            end
            while true
                length_cache = length(active_list)
                for _ in 1:length_cache
                    i = popfirst!(active_list)
                    if data[i] / w[i] > pivot
                        push!(active_list, i)
                    else
                        s1 = s1 - data[i] * w[i]
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
    wcondat_s(data, w, b)

Weighted simplex projection based on serial Condat's method
"""
function wcondat_s(data::Array{Float64,1}, w::Array{Float64,1}, b::Real)::AbstractVector
    active_list, s1, s2 = wfilter(data, w, b)
    return wcheckL(active_list, s1, s2, data, w, b)
end

"""
    wcondat_s(data, w, b, numthread)

Weighted simplex projection based on parallel Condat's method
"""
function wcondat_p(
    data::Array{Float64,1}, w::Array{Float64,1}, b::Real, numthread
)::AbstractVector
    active_list, s1, s2 = parallel_wfilter(data, w, b, numthread)
    return wcheckL(active_list, s1, s2, data, w, b)
end
