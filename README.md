# NewtonCQK

Implementation a semi-smooth Newton method for the Continuous Quadratic
Knapsack problem, with specialized implemetations for projecting onto a
Simplex and $\ell_1$-ball.

It is authored by Paulo J. S. Silva and Leonardo D. Secchin.


## Installation and use

`]add https://github.com/pjssilva/NewtonCQK.jl`

The package solves problems of the form

$$\min_x \frac{1}{2}x^tDx - a^t \quad \text{s.t.} \quad b^tx = r, \ l \leq x \leq u,$$

where $D$ is a positive diagonal matrix, $b$ is a vector with positive
entries, and $l\leq u$ (some bounds may be $\pm \infty$). Problems are stored
in the `CQKProblem` structure; type `?CQKProblem` for details. For solver
instructions, type `?cqk!` and `?cqk`.

To project a vector onto a simplex, see `?simplex_proj!` and `?simplex_proj`.
These functions return solutions as dense vectors. For sparse vectors, use
`spsimplex_proj`. There are similar functions for projecting onto an
$\ell_1$-ball: `l1ball_proj!`, `l1ball_proj` and `spl1ball_proj`. All these
functions receive the vector to be projected and $r$ as input parameters.

Input vectors, including those in the `CQKProblem` structure, can be of
type `CuVector`, provided by the `CUDA.jl` package. In this case, the solvers
run on the GPU.


## Funding

This research was supported by the São Paulo Research Foundation (FAPESP)
(grants 2013/07375-0, 2023/08706-1 and 2024/12967-8) and the National Council
for Scientific and Technological Development (CNPq) (grants 302520/2025-2 and
312394/2023-3), Brazil.


## How to cite

If you use this code in your publications, please cite us. For now, you
can cite the preprint:

[*Secchin, Silva. Parallel Newton methods for the continuous quadratic knapsack problem: A Jacobi and Gauss-Seidel tale. 2025*]()
