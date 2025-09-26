#condat.jl

using Base.Threads #for @parallel macro
using BangBang #for append!!() function
using SparseArrays #for sparse vector

"""
    serial_filter(data, b)

Filter method in Condat's method, which is a preprocessing method.
"""
function serial_filter(data::Array{Float64,1}, b::Real=1)::Array{Float64,1}
    #initialize active list
    active_list = Float64[data[1]]
    #initialize the length of active list
    length_list = 1
    #initialize pivot
    pivot = active_list[1] - b
    #initialize waiting list
    wait_list = Float64[]
    for i in 2:length(data)
        #remove inactive terms
        if data[i] > pivot
            #update pivot
            length_list += 1
            pivot = pivot + (data[i] - pivot) / (length_list)
            if pivot > data[i] - b
                push!(active_list, data[i])
            else
                #if pivot is too large, push previous list to waiting list
                append!!(wait_list, active_list)
                active_list = [data[i]]
                pivot = data[i] - b
                length_list = 1
            end
        end
    end
    #reuse some terms from waiting list
    for x in wait_list
        if x > pivot
            push!(active_list, x)
            length_list += 1
            pivot = pivot + (x - pivot) / (length_list)
        end
    end
    #output active list
    return active_list
end

"""
    checkL(active_list, b)

Project vector (with elements from active_list) onto a simplex with the scaling
factor b based on the second part of Condat's method, and it returns τ of the
projection
"""
function checkL(active_list::Array{Float64,1}, b::Real=1)::Float64
    #initialize pivot
    pivot = (sum(active_list) - b) / length(active_list)
    while true
        #record length
        local length_list = length(active_list)
        #check all terms in current active_list
        for i in 1:length_list
            x = popfirst!(active_list)
            if x <= pivot
                pivot = pivot + (pivot - x) / length(active_list)
            else
                push!(active_list, x)
            end
        end
        #check termination condition
        if length_list <= length(active_list)
            return pivot
        end
    end
end

"""
    parallel_filter(data, b, numthread, spa)

Split vector into "numthread" subvectors, and project these subvectors
synchronously on different threads. Note that, such projection can be terminated
early with sparse rate "spa" (see "michelot.jl" for more details).
"""
function parallel_filter(
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
            #initialize pivot
            local pivot = data[st] - b
            #initialize length
            local length_list = 1
            #initialize active list and waiting list
            local local_list = Float64[data[st]]
            local wlist = Float64[]
            sizehint!(local_list, max(1024, (en - st + 1) ÷ 50))
            sizehint!(wlist, max(1024, (en - st + 1) ÷ 50))
            for i in (st + 1):en
                if data[i] > pivot
                    pivot += (data[i] - pivot) / (length_list += 1)
                    if pivot > data[i] - b
                        push!(local_list, data[i])
                    else
                        append!!(wlist, local_list)
                        local_list = Float64[data[i]]
                        pivot = data[i] - b
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
