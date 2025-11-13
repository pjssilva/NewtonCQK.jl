#!/usr/bin/bash

juliacmd="julia-1.10"

# Random
rm -f results.jld2
for t in 1 2 4 8 16 24 48; do
    nice -n 0 $juliacmd --project -t $t benchmark.jl -continue true -svm false --svm_tune false --random true --bp false --cuda 64 ---nreps 20
done

# Basis pursuit
rm -f results_bp.jld2
for t in 1 2 4 8 16 24 48; do
    nice -n 0 $juliacmd --project -t $t benchmark.jl --continue true --svm false --svm_tune false --random false --bp true
done

# SVM
rm -f results_svm.jld2
for t in 1 2 4 8 16 24 48; do
    nice -n 0 $juliacmd --project -t $t benchmark.jl --continue true --svm true --svm_tune false --random false --bp false
done
