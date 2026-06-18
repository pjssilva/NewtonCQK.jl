# Third party soruce code

This directory containts third party soruce code used for validation and
benchmark. This third-party code is distributed under its original terms. See
the respective `LICENSE` or `COPYING` files.

## Condat's code

You shoud download the file `condat_simplexproj.c` from [Laurent
Condat](https://lcondat.github.io/) webpage and place it in the `condat`
subdirectory. After that, in this directory, run `make` to generate the
respective library that will be used in the benchmark.

## Parallel Simplex Projection

The `Parallel Simplex Projection` from Yongzheng Dai and Chen Chen can be
obtaned clonning the GitHub repository
[git@github.com:foreverdyz/Parallel-Simplex-Projection.git](git@github.com:foreverdyz/Parallel-Simplex-Projection.git).
Just type
```bash
git clone git@github.com:foreverdyz/Parallel-Simplex-Projection.git
```
In the current diretory to get it.

## Semismooth Newton method for the Continuous Quadratic Knapsack

For convenience, we distribute the code from Cominetti, Mascarenhas, and
Silva. But you have to compile it. Go to the directory
`quadratic_knapsack_source/lib` and type `make` to compile it.

## Hager and Zhang's PPROJ method

The [`PPROJ` method](https://doi.org/10.1137/15M102825X) for projecting onto a
polyhedron, from William Hager and Hongchao Zhang, can be obtained from the
Hager' page. PPROJ v1.0 requires Tim Davis' SuiteSparse (v4.4.7) and George
Karypis' Metis (v4.0). For convenience, we included the script
`pproj/download_and_extract.sh` that download and extract all necessary stuff.
After that, go to the directory `pproj` and type `make`. Note that you need to
install the packages `cmake` and `libmpfr-dev` (or equivalent) in your system
before.

## Commercial solvers

If IBM CPLEX, GUROBI and/or Hexaly are installed on your system, you can compile
the corresponding interface. To do so, define the environment variables
`cplex_path`, `gurobi_path` and `hexaly_path` to point to the respective
installation directories. Then, type `make` from the appropriate interface
directory (`cplex`, `gurobi` or `hexaly`).
