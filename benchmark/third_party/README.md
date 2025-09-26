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

For convinience, we distribute the code from Cominetti, Mascarenhas, and
Silva. But you have to compile it. Go to the directory
`quadratic_knapsack_source/lib` and type `make` to compile it.
