# Description of Scripts

## Projecton onto a weighted simplex

wsortscan.jl, wmichelot.jl, and wcondat.jl provide 6 projection methods (for a weighted simplex) based on serail sort and scan (wsortscan_s), parallel sort and scan (wsortscan_p), serial Michelot's method (wmichelot_s), parallel Michelot's method (wmichelot_p), serial Condat's method (wcondat_s) and parallel Condat's method ((wcondat_p)). 

### Example to use them

```julia
julia> include("wsortscan.jl") #load 7 methods from the script
julia> using Random, Distributions, Base.Threads
julia> data = rand(N(0, 1), 1_000_000) #generate a random input vector i.i.d. $N(0, 1)$ with size of $10^6$
julia> w = rand(1_000_000) #generate a random weight vector i.i.d. $U[0, 1]$ with size of $10^6$
julia> wsortscan_s(data, w, 1) #use serial sort and scan method to project input vector data onto a weighted simplex with scaling factor 1
julia> include("wmichelot.jl")
julia> wmichelot_p(data, w, nthreads()) #use parallel Michelot's method to project input vector data onto a weighted simplex with scaling factor 1, and nthreads() return the number of available threads
```

## Projecton onto a weighted l1 ball

wl1ball.jl provides 6 projection methods (for a weighted l1 ball) based on serail sort and scan (wball_sortscan_s), parallel sort and scan (wball_sortscan_p), serial Michelot's method (wball_michelot_s), parallel Michelot's method (wball_michelot_p), serial Condat's method (wball_condat_s) and parallel Condat's method (wball_condat_p).

### Example to use them

```julia
julia> include("wl1ball.jl")
julia> using Random, Distributions, Base.Threads
julia> data = rand(N(0, 1), 1_000_000) #generate a random input vector i.i.d. $N(0, 1)$ with size of $10^6$
julia> w = rand(1_000_000) #generate a random weight vector i.i.d. $U[0, 1]$ with size of $10^6$
julia> wball_sortscan_s(data, w, 1) #use serial sort and scan method to project input vector data onto a weighted l1 ball with scaling factor 1
julia> wball_michelot_p(data, w, nthreads()) #use parallel Michelot's method to project input vector data onto a weighted l1 ball with scaling factor 1, and nthreads() return the number of available threads
```
