#l1ball_wrap.jl

using Base.Threads #for @parallel macro
using BangBang #for append!!() function
using SparseArrays #for sparse vector

include("simplex_wrap.jl")

"""
    l1ball_scan(data, pivot)

Given piovt for simplex projection of abs.(data) (input vector) onto the simplex,
    l1ball_scan returns sign.(data).*max.(abs.(data).-pivot, 0) with sparse
    structure.
"""
function l1ball_scan(data::Array{Float64,1}, pivot::Real)
    #initialize two sets
    I_p = Int[] #active index set
    V = Float64[] #value set
    for i in 1:length(data)
        if abs(data[i]) > pivot
            push!(I_p, i)
            if data[i] >= 0
                push!(V, data[i] - pivot)
            else
                push!(V, data[i] + pivot)
            end
        end
    end
    return sparsevec(I_p, V, length(data))
end

"""
    l1ball_pscan(data, pivot, numthread)

Similar to l1ball_scan(), but using multiple threads in
    sign.(data).*max.(abs.(data).-pivot, 0).
"""
function l1ball_pscan(data::Array{Float64,1}, pivot::Real, numthread::Int)::AbstractVector
    I_p = Int[]
    V = Float64[]
    # return sparsevec(I_p, V, length(data))
    width = ceil(Int, length(data) / numthread)
    spl = SpinLock()
    @threads for i in 1:width:length(data)
        let
            local st = i
            local en = min(i + width - 1, length(data))
            local local_I = Int[]
            local local_V = Float64[]
            for j in st:en
                local x = abs(data[j]) - pivot
                if x > 0
                    push!(local_I, j)
                    if data[j] >= 0
                        push!(local_V, x)
                    else
                        push!(local_V, -x)
                    end
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

"""
    l1ball_sortscan_s(data, b)

l1 ball projection based on serial sort and scan simplex method
"""
function l1ball_sortscan_s(data::Array{Float64,1}, b::Real)::AbstractVector
    d = abs.(data)
    pivot = sortscan_pivot(d, b)
    return l1ball_scan(data, pivot)
end

"""
    l1ball_sortscan_p(data, b, numthread)

l1 ball projection based on parallel sort ans scan simplex method
"""
function l1ball_sortscan_p(data::Array{Float64,1}, b::Real, numthread::Int)::AbstractVector
    d = abs.(data)
    pivot = sortscan_parallel_pivot(d, b)
    return l1ball_pscan(data, pivot, numthread)
end

"""
    l1ball_sortPscan_p(data, b, numthread)

l1 ball projection based on parallel sort and partial scan simplex method
"""
function l1ball_sortPscan_p(data::Array{Float64,1}, b::Real, numthread::Int)::AbstractVector
    d = abs.(data)
    d_after_sort = sort(d; alg=ThreadsX.MergeSort, rev=true)
    #get the pivot
    pivot = parallel_partial_scan(d_after_sort, b)
    return l1ball_pscan(data, pivot, numthread)
end

"""
    l1ball_michelot_s(data, b)

l1 ball projection based on serial Michelot's method
"""
function l1ball_michelot_s(data::Array{Float64,1}, b::Real)::AbstractVector
    d = abs.(data)
    pivot = serial_scan(d, b)
    return l1ball_scan(data, pivot)
end

"""
    parallel_scan_l1ball(data, b, numhtread, spa)
"""
function parallel_scan_l1ball(
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
            local st = (i - 1) * width + 1
            if i == numthread
                local en = length(data)
            else
                local en = i * width
            end
            local local_list = michelot_local_scan(abs.(data[st:en]), width, b, spa)
            #reduce with locking
            lock(spl)
            append!!(active_list, local_list)
            unlock(spl)
        end
    end
    return active_list
end

"""
    l1ball_michelot_p(data, b, numthread, spa)

l1 ball projection based on parallel Michelot's simplex method
"""
function l1ball_michelot_p(
    data::Array{Float64,1}, b::Real, numthread::Int, spa::Float64
)::AbstractVector
    active_list = parallel_scan_l1ball(data, b, numthread, spa)
    pivot = serial_scan(active_list, b)
    return l1ball_pscan(data, pivot, numthread)
end

"""
    l1ball_condat_s(data, b)

l1 ball projection based on serial Condat's simplex method
"""
function l1ball_condat_s(data::Array{Float64,1}, b::Real)::AbstractVector
    d = abs.(data)
    active_list = serial_filter(d, b)
    pivot = checkL(active_list, b)
    return l1ball_scan(data, pivot)
end

"""
    parallel_filter_l1ball(data, b, numthread, spa)
"""
function parallel_filter_l1ball(
    data::Array{Float64,1}, b::Real, numthread::Int, spa::Float64
)::Array{Float64,1}
    #the length for subvectors
    width = floor(Int, length(data) / numthread)
    #lock global value
    spl = SpinLock()
    #initialize a global list
    glist = Float64[]
    @threads for id in 1:numthread
        let
            #determine start and end position for subvectors
            local st = (id - 1) * width + 1
            if id == numthread
                local en = length(data)
            else
                local en = id * width
            end
            #local d = abs.(data[st:en])
            #initialize pivot
            local pivot = abs(data[st]) - b
            #initialize length
            local length_list = 1
            #initialize active list and waiting list
            local local_list = Float64[abs(data[st])]
            local wlist = Float64[]
            #local length_cache = en-st+1
            for i in (st + 1):en
                x = abs(data[i])
                if x > pivot
                    pivot += (x - pivot) / (length_list += 1)
                    if pivot > x - b
                        push!(local_list, x)
                    else
                        append!!(wlist, local_list)
                        local_list = Float64[x]
                        pivot = x - b
                        length_list = 1
                    end
                end
            end
            #reuse terms from waiting list
            for x in wlist
                if x > pivot
                    push!(local_list, x)
                    pivot += (x - pivot) / (length_list += 1)
                end
            end

            while length_list > width * spa
                #initialize the summation for this iteration
                local local_sum = -b
                #check all terms in current local_active_list
                for j in 1:length_list
                    y = popfirst!(local_list)
                    if y <= pivot
                        #update pivot
                        pivot = pivot + (pivot - y) / (length(local_list))
                    else
                        push!(local_list, y)
                        local_sum += y
                    end
                end
                #check termination condition
                if length_list <= length(local_list)
                    break
                end
                #update length in every iteration
                length_list = length(local_list)
            end

            #reduce with locking
            lock(spl)
            append!!(glist, local_list)
            unlock(spl)
        end
    end
    return glist
end

"""
    l1ball_condat_p(data, b, numthread, spa)

lq ball projection based on parallel Condat's simplex method
"""
function l1ball_condat_p(
    data::Array{Float64,1}, b::Real, numthread::Int, spa::Float64
)::AbstractVector
    active_list = parallel_filter_l1ball(data, b, numthread, spa)
    pivot = checkL(active_list, b)
    return l1ball_pscan(data, pivot, numthread)
end
