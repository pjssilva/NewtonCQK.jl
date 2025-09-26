#!/usr/bin/bash

rm -f results.jld2

for t in 1 2 4 8 16 24 48; do
    nice -n 0 julia -t $t benchmark.jl --continue true --nreps 20
done
