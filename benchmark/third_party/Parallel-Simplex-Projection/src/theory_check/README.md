# Descript of scripts

## average_active.jl

This script calculates the number of active terms in simplex projection for different sizes $n$ of the input vector, where $n$ is between $10^6$ and $10^7$. Then compare these numbers with series $\sqrt{2n}$. To get the plot result, see the following example.

```julia
julia> include("average_active.jl")
```
![average_active.png](/results/average_active.png)

## filter_result.jl

This script calculates the number of active terms after applying Filter technique the input vector, where the input vector has different sizes between $10^6$ and $10^7$. Then compare these numbers with series $(2.2n)^{\frac{2}{3}}$. To get the plot result, see the following example.

```julia
julia> include("filter_result.jl")
```
![filter_result.png](/results/filter_result.png)

## michelot_loop.jl

This scripts calculates the number of active terms after every loop in Michelot's method, and the input vector has a size of $10^6$. Then compare these numbers with series $(10^6)^{2^t}$, where $t$ is the number of iterations. To get the plot result, see the following example.

```julia
julia> include("michelot_loop.jl")
```
![filter_result.png](/results/michelot_loop.png)
