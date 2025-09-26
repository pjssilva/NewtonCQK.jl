#michelot.jl

using Base.Threads #for @parallel macro
using BangBang #for append!!() function
using SparseArrays #for sparse vector

"""
    serial_scan(d, b)

serial_scan() returns τ for the projection of d onto simplex with scaling
factor b based on Michelot method.
"""
function serial_scan(d::Array{Float64,1}, b::Real)::Float64
    #initialize the pivot
    pivot = (sum(d) - b) / length(d)
    #remove inactive terms in each iteration
    while true
        #record the current length
        length_cache = length(d)
        #initailize the summation for each iteration
        summation = -b
        #check all terms in current d
        for i in 1:length_cache
            x = popfirst!(d)
            #remove inactive terms
            if x > pivot
                push!(d, x)
                summation += x
            end
        end
        #check the condition for termination
        if length_cache == length(d)
            return pivot
        end
        #update pivot
        pivot = summation / length(d)
    end
end

"""
    parallel_scan(data, b, numthread, spa)

parallel_scan() splits input vector (data) and broadcasts to numthread (number
of cores) cores, and then it projects subvector onto simplex with scaling factor
b. Reduce active terms from each core to the main core, and return the reduced
vector. Note that, in the projection of subvectors, the projection can be
terminated early with sparse rate "spa", e.g. stop when
# of active terms / size of subvector < spa
"""
function parallel_scan(
    data::Array{Float64,1}, b::Real, numthread::Int, spa::Float64
)::Array{Float64,1}
    #the length for subvectors
    width = floor(Int, length(data) / numthread)
    #lock global value
    spl = SpinLock()
    #initialize the final list
    active_list = Float64[]
    @threads for i in 1:numthread
        #do serial scan for subvectors here
        let
            #determine start and end position for subvectors
            #here using "local" to prevent some glitchs in parallel library
            local st = (i - 1) * width + 1
            if i == numthread
                local en = length(data)
            else
                local en = i * width
            end
            #same to previous "local"
            #data[st:en] is a copy, which prevent changing data
            local local_list = michelot_local_scan(data[st:en], width, b, spa)
            #reduce with locking
            lock(spl)
            append!!(active_list, local_list)
            unlock(spl)
        end
    end
    return active_list
end

"""
    michelot_local_scan(d, width, b, spa)

A sub-function of parallel_scan(), which projects the subvector onto the
simplex and returns active terms.
"""
function michelot_local_scan(
    d::Array{Float64,1}, width::Int, b::Real, spa::Float64
)::Array{Float64,1}
    #record the current length
    length_cache = length(d)
    #initailize the pivot
    pivot = (sum(d) - b) / length_cache
    while true
        #initialize the summation for this interation
        summation = -b
        #check all terms in the current d
        for i in 1:length_cache
            if (x = popfirst!(d)) > pivot
                push!(d, x)
                summation += x
            end
        end
        #check conditions for the termination
        if length(d) == length_cache
            return d
        end
        length_cache = length(d)
        #early termination condition
        if length_cache / width < spa
            return d
        end
        pivot = summation / length_cache
    end
end

"""
    scan(data, pivot)
Returns max.(data.-pivot, 0) with sparse structure
"""
function scan(data::Array{Float64,1}, pivot::Real)::AbstractVector
    #initialize two sets
    I_p = Int[] #active index set
    V = Float64[] #value set
    for i in 1:length(data)
        if data[i] > pivot
            push!(I_p, i)
            push!(V, data[i] - pivot)
        end
    end
    return sparsevec(I_p, V, length(data))
end

"""
    pscan(data, pivot, numthread)

Similar to scan(), but using multiple threads in max.(data[i]-pivot, 0).
"""
function pscan(data::Array{Float64,1}, pivot::Real, numthread::Int)::AbstractVector
    I_p = Int[]
    V = Float64[]
    width = ceil(Int, length(data) / numthread)
    spl = SpinLock()
    @threads for i in 1:width:length(data)
        let
            local st = i
            local en = min(i + width - 1, length(data))
            local local_I = Int[]
            local local_V = Float64[]
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
    return sparsevec(I_p, V, length(data))
end
