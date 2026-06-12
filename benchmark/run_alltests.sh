#!/usr/bin/bash

# Julia executable
juliacmd="julia-1.10"
# juliacmd="julia +1.10.9"

# Threads
threads="1 2 4 8 16 24 48"

# Paths of commercial solvers
cplex_path="/opt/ibm/ILOG/CPLEX_Studio2211/cplex"
gurobi_path="/opt/gurobi1302/linux64"
hexaly_path="/opt/hexaly_14_5"

# Compile third-party software
echo "Compiling third-party software..."
(cd third_party/condat/ && make)
(cd third_party/quadratic_knapsack_source/lib/ && make)
if [ -d $cplex_path ]; then
    (cd third_party/cplex/ && make)
fi
if [ -d $gurobi_path ]; then
    (cd third_party/gurobi/ && make)
fi
if [ -d $hexaly_path ]; then
    (cd third_party/hexaly/ && make)
fi

# Set 1 thread for BLAS (for third-party C code)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Random
for t in $threads; do
    nice -n 0 $juliacmd --project -t $t random/runtests.jl --continue true --nreps 20 --cuda 64
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
