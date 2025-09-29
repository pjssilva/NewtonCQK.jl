#!/usr/bin/bash

rm -f results.jld2

for t in 1 2 4 8 16 24 48; do
    nice -n 0 julia-1.10 --project -t $t benchmark.jl --cuda 64 --continue true --nreps 20
done
