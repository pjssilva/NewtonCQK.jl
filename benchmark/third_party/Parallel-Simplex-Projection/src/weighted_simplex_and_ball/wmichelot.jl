#wmichelot.jl

using Base.Threads
using BangBang
using SparseArrays

"""
    wmichelot_s(data, w, b)

Weighted simplex projection based on serial Michelot's method
"""
function wmichelot_s(data::Array{Float64,1}, w::Array{Float64,1}, b::Real)::AbstractVector
    let
        s1 = data' * w
        s2 = w' * w
        pivot = (s1 - b) / s2
        active_list = [i for i in 1:length(data)]
        while true
            length_cache = length(active_list)
            for _ in 1:length_cache
                i = popfirst!(active_list)
                if data[i] / w[i] > pivot
                    push!(active_list, i)
                else
                    s1 = s1 - data[i] * w[i]
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
            push!(value_list, data[j] - w[j] * pivot)
        end
        return sparsevec(active_list, value_list, length(data))
    end
end

"""
    wmichelot_p(data, w, b, numthread)

Weighted simplex projetion based on parallel Michelot's method
"""
function wmichelot_p(data::Array{Float64,1}, w::Array{Float64,1}, b::Real, numthread::Int)
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
                local active_list = [i for i in st:en]
                local s1 = data[st:en]' * w[st:en]
                local s2 = w[st:en]' * w[st:en]
                local pivot = (s1 - b) / (s2)
                while true
                    length_cache = length(active_list)
                    for _ in 1:length_cache
                        i = popfirst!(active_list)
                        if data[i] / w[i] > pivot
                            push!(active_list, i)
                        else
                            s1 = s1 - data[i] * w[i]
                            s2 = s2 - w[i]^2
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
                if data[i] / w[i] > pivot
                    push!(glist, i)
                else
                    gs1 = gs1 - data[i] * w[i]
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
            push!(value_list, data[j] - w[j] * pivot)
        end
        return sparsevec(glist, value_list, length(data))
    end
end
