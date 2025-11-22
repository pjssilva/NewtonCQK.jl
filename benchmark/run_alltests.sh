#!/usr/bin/bash

# juliacmd="julia-1.10"
juliacmd="julia +1.10.9"
threads="1 2 4 8 16 24 48"

# Random
for t in $threads; do
    nice -n 0 $juliacmd --project -t $t random/runtests.jl --continue true #--nreps 20 --cuda 64
done

# Basis pursuit
for t in $threads; do
    nice -n 0 $juliacmd --project -t $t basis_pursuit/runtests.jl --continue true
done

# SVM
nice -n 0 $juliacmd --project svm/download_datasets.jl
for t in $threads; do
    nice -n 0 $juliacmd --project -t $t svm/runtests.jl --continue true
done

# Results
nice -n 0 $juliacmd --project results/results.jl
