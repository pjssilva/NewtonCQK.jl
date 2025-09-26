#parity_polytope.jl

using Base.Threads
using ThreadsX

include("simplex_and_l1ball/simplex_wrap.jl")

"""
    parity_s(f, data)

Parity Polytope projection based on serial method, f
"""
function parity_s(f::Function, data::Array{Float64,1})::AbstractVector
    f_list = zeros(length(data))
    for i in 1:length(data)
        f_list[i] = data[i] < 0 ? 0 : 1
    end
    if sum(f_list) % 2 == 0
        i = findmin(abs.(data))[2]
        f_list[i] = 1 - f_list[i]
    end
    v = zeros(length(data))
    d = zeros(length(data))
    for i in 1:length(data)
        v[i] = data[i] * (-1)^(f_list[i])
        d[i] = max(min(v[i], 0.5), -0.5)
    end
    if sum(d) >= (1 - length(data) / 2)
        for i in 1:length(data)
            d[i] = max(min(data[i], 0.5), -0.5)
        end
        return d
    else
        x = Array(f(v, 1))
        for i in 1:length(data)
            x[i] = (x[i] - 0.5) * (-1)^(f_list[i])
        end
        return x
    end
end

"""
    parity_p(f, data, spa)

Parity Polytope projection based on parallel method, f
"""
function parity_p(f::Function, data::Array{Float64,1}, spa::Float64)::AbstractVector
    f_list = zeros(length(data))
    @threads for i in 1:length(data)
        f_list[i] = data[i] < 0 ? 0 : 1
    end
    if sum(f_list) % 2 == 0
        i = findmin(abs.(data))[2]
        f_list[i] = 1 - f_list[i]
    end
    v = zeros(length(data))
    d = zeros(length(data))
    @threads for i in 1:length(data)
        @inbounds v[i] = data[i] * (-1)^(f_list[i])
        @inbounds d[i] = max(min(v[i], 0.5), -0.5)
    end
    if ThreadsX.sum(d) >= (1 - length(data) / 2)
        @threads for i in 1:length(data)
            @inbounds d[i] = max(min(data[i], 0.5), -0.5)
        end
        return d
    else
        x = Array(f(v, 1, nthreads(), spa))
        @threads for i in 1:length(data)
            @inbounds x[i] = (x[i] - 0.5) * (-1)^(f_list[i])
        end
        return x
    end
end

"""
    parity_p_sortscan(f, data)

Parity Polytope projection based on parallel Sort and (partial) Scan method
"""
function parity_p_sortscan(f::Function, data::Array{Float64,1})::AbstractVector
    f_list = zeros(length(data))
    @threads for i in 1:length(data)
        f_list[i] = data[i] < 0 ? 0 : 1
    end
    if sum(f_list) % 2 == 0
        i = findmin(abs.(data))[2]
        f_list[i] = 1 - f_list[i]
    end
    v = zeros(length(data))
    d = zeros(length(data))
    @threads for i in 1:length(data)
        @inbounds v[i] = data[i] * (-1)^(f_list[i])
        @inbounds d[i] = max(min(v[i], 0.5), -0.5)
    end
    if ThreadsX.sum(d) >= (1 - length(data) / 2)
        @threads for i in 1:length(data)
            @inbounds d[i] = max(min(data[i], 0.5), -0.5)
        end
        return d
    else
        x = Array(f(v, 1, nthreads()))
        @threads for i in 1:length(data)
            @inbounds x[i] = (x[i] - 0.5) * (-1)^(f_list[i])
        end
        return x
    end
end

"""
    parity_polytope_prepare(data)

Preprocess data for testing runtim of simplex projection in parity polytope
    projection
"""
function parity_polytope_prepare(data::Array{Float64,1})::AbstractVector
    let
        f = zeros(length(data))
        for i in 1:length(data)
            f[i] = data[i] < 0 ? 0 : 1
        end
        if sum(f) % 2 == 0
            i = findmin(abs.(data))[2]
            f[i] = 1 - f[i]
        end
        v = zeros(length(data))
        for i in 1:length(data)
            v[i] = data[i] * (-1)^(f[i])
        end
        return v
    end
end
