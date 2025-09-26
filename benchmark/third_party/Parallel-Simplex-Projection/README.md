# Parallel Simplex Projection

This archive is under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in [Sparsity-Exploiting Distributed Projections onto a Simplex](https://doi.org/10.1287/ijoc.2022.0328) by Yongzheng Dai and Chen Chen.  See also ([https://arxiv.org/abs/2111.07701](https://arxiv.org/abs/2204.08153)).

## Background 

Projecting a vector onto a simplex is a well-studied problem that arises in a wide range of optimization problems.  Numerous algorithms have been proposed for determining the projection; however, the primary focus of the literature has been on serial algorithms. We present a parallel method that decomposes the input vector and distributes it across multiple processors for local projection. Our method is especially effective when the resulting projection is highly sparse; which is the case, for instance, in large-scale problems with i.i.d. entries. Moreover, the method can be adapted to parallelize a broad range of serial algorithms from the literature. We fill in theoretical gaps in serial algorithm analysis, and develop similar results for our parallel analogues. Numerical experiments conducted on a wide range of large-scale instances, both real-world and simulated, demonstrate the practical effectiveness of the method.  

## Description

We implemented the following parallel projections based on the simplex projection:
- projection onto a simplex (see folder [simplex_and_l1ball](/src/simplex_and_l1ball));
- projection onto an $\ell_1$ ball (see folder [simplex_and_l1ball](/src/simplex_and_l1ball));
- projection onto a weighted simplex and a weighted $\ell_1$ ball (see folder [weighted_simplex_and_l1ball](/src/weighted_simplex_and_l1ball));
- projection onto a parity polytope (see file [parity_polytope.jl](/src/parity_polytope.jl));

We also conduct the following experiments to test our algorithms:
- Runtime fairness test: serial method vs. parallel method in 1 core (see file [fairness_test.jl](/src/fairness_test.jl));
- Unit test: check our parallel methods will return the same (and correct) results as serial methods (see file [unit_test.jl](/src/unit_test.jl)); 
- Some theories check (see our [paper](https://doi.org/10.1287/ijoc.2022.0328) for details and find code in folder [theory_check](/src/theory_check));
- Benchmark runtime for the simplex projection with input vectors $d$ in multiple sizes and distributions (see file [simplex_runtime_benchmark.jl](/src/simplex_runtime_benchmark.jl));
- Benchmark runtime for the $\ell_1$ ball projection with the input vector $d$ in $N(0,1)$ and size of $10^8$ (see file in [l1ball_runtime_benchmark.jl](/src/l1ball_runtime_benchmark.jl));
- Benchmark runtime for the weighted simplex projection with the input vector $d$ in $N(0,1)$ and size of $10^8$ and the weight $w$ in $U[0,1]$ (see file [wsimplex_runtime_benchmark.jl](/src/wsimplex_runtime_benchmark.jl));
- Benchmark runtime for the weighted $\ell_1$ ball projection with the same setting as the weighted simplex projection (see file [wl1ball_runtime_benchmark.jl](/src/wl1ball_runtime_benchmark.jl));
- Benchmark runtime for the parity polytope projection with the input vector $d$ in $U[1,2]$ and size of $10^8-1$ (see file [paritypolytope_runtime_benchmark.jl](/src/paritypolytope_runtime_benchmark.jl));
- Implemented Lasson method for two real-world datasets: [kdd2010](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2010%20(algebra)) and [kdd2012](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2012) (see files [real_data_kdd10.jl](/src/real_data_kdd10.jl) and [real_data_kdd12.jl](/src/real_data_kdd12.jl)).

## Language

The code is written in the [Julia programming language](https://julialang.org). Please visit the website for an installation guide. 

## Required packages 

To run the content of scripts in [src](/src), the following Julia packages are required: ThreadsX, BangBang, BenchmarkTools, Random, Distributions, SparseArrays.

To install a package simply run

```julia
pkg> add PACKAGE_NAME    # Press ']' to enter the Pkg REPL mode.
```

## Replicating

All results appearing in the paper are collected in [results](results). Information on how to reproduce these results can be found in the [README.md (in results)](results/README.md) or see the subsection plot_graphs.py in the [README.md (in src)](/src/README.md) for a simple example. Note that, since the two real datasets [kdd2010](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2010%20(algebra)) and [kdd2012](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2012) are too large to upload, we split them into some small files. You can combine these subfiles by running [data_combine.jl](data/data_combine.jl)or directly download them as the tutorial in [README.md (in results)](results/README.md) and move them to [src](/src) before you run [real_data_kdd10.jl](/src/real_data_kdd10.jl) and [real_data_kdd12.jl](/src/real_data_kdd12.jl) to reproduce results of real-world datasets. 

## Data 

All real-world data that was used is available in [data](data). Other data are generated randomly (with random seed 12345) based on some distributions.

Here is an example to generate the random data

```julia
julia> using Distributions, Random
julia> Random.seed!(12345); data = rand(N(0,1), 1_000_000) #generate a vector with size of 10^6 from the standard normal distribution
```

## Licenses

We have [MIT License](LICENSE) for all code and [BSD 3-Clause License](data/LICENSE) for real-world data we used.
