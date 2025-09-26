#simplex.jl

include("sort_scan.jl")
"""
    sortscan_s(data, b)

Simplex projection method: serial sort and scan
"""
function sortscan_s(data::Array{Float64,1}, b::Real)::AbstractVector
    pivot = sortscan_pivot(data, b)
    return scan(data, pivot)
end

"""
    sortscan_p(data, b, numthread)

Simplex projection method: parallel sort and scan
"""
function sortscan_p(data::Array{Float64,1}, b::Real, numthread::Int)::AbstractVector
    pivot = sortscan_parallel_pivot(data, b)
    return pscan(data, pivot, numthread)
end

"""
    sortPscan_p(data, b, numthread)

Simplex projection method: parallel sort and partial scan
"""
function sortPscan_p(data::Array{Float64,1}, b::Real, numthread::Int)::AbstractVector
    #using parallel mergesort here
    d_after_sort = sort(data; alg=ThreadsX.MergeSort, rev=true)
    #get the pivot
    pivot = parallel_partial_scan(d_after_sort, b)
    return pscan(data, pivot, numthread)
end

include("michelot.jl")
"""
    michelot_s(data, b)

Simplex projection method: Michelot's method
"""
function michelot_s(data::Array{Float64,1}, b::Real)::AbstractVector
    #here we need to make a copy of data, otherwise the algorithm will change data
    pivot = serial_scan(data[1:end], b)
    return scan(data, pivot)
end

"""
    michelot_p(data, b, numthread, spa)

Simplex projection method: Parallel Michelot's method
"""
function michelot_p(
    data::Array{Float64,1}, b::Real, numthread::Int, spa::Float64
)::AbstractVector
    #get all active elements from muliple cores
    active_list = parallel_scan(data, b, numthread, spa)
    #get τ
    pivot = serial_scan(active_list, b)
    return pscan(data, pivot, numthread)
end

include("condat.jl")
"""
    condat_s(data, b)

Simplex projection method: Condat's method
"""
function condat_s(data::Array{Float64,1}, b::Real=1)::AbstractVector
    #filter technique
    active_list = serial_filter(data, b)
    #get τ
    pivot = checkL(active_list, b)
    return scan(data, pivot)
end

"""
    condat_p(data, b, numthread, spa)

Simplex projection method: Parallel Condat's method
"""
function condat_p(
    data::Array{Float64,1}, b::Real, numthread::Int, spa::Float64
)::AbstractVector
    #parallel filter and scan
    active_list = parallel_filter(data, b, numthread, spa)
    #get τ
    pivot = checkL(active_list, b)
    return pscan(data, pivot, numthread)
end
