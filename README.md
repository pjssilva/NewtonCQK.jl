# NewtonCQK

Implementation a semi-smooth Newton method for the Continuous Quadratic
Knapsack problem, with specialized implemetations for projecting onto a
Simplex and $\ell_1$-ball.

It is authored by Paulo J. S. Silva and Leonardo D. Secchin.


## Installation and use

To install the package, run the following command in Julia:

`]add https://github.com/pjssilva/NewtonCQK.jl`

The package solves problems of the form

$$\min_x \frac{1}{2}x^tDx - a^t \quad \text{s.t.} \quad b^tx = r, \ l \leq x \leq u,$$

where $D$ is a positive diagonal matrix, $b$ is a vector with positive
entries, and $l\leq u$ (some bounds may be $\pm \infty$). Problems are stored
in the `CQKProblem` structure; type `?CQKProblem` for details. For
instructions on using the solver, type `?cqk!` or `?cqk`.

To project a vector onto a simplex, see `?simplex_proj!` or `?simplex_proj`.
These functions return solutions as dense vectors. For sparse vectors, use
`spsimplex_proj`. Similar functions are available for projection onto an
$\ell_1$-ball: `l1ball_proj!`, `l1ball_proj` and `spl1ball_proj`. All these
functions receive the vector to be projected and $r$ as input parameters.

Input vectors, including those in the `CQKProblem` structure, can be of
type `CuVector`, provided by the `CUDA.jl` package. In this case, the solvers
run on the GPU.

## Examples

Solving a CQK problem:
```
using NewtonCQK

n = 10^5
d = ones(n)
a = rand(n)
b = rand(n)
r = 10.0
l = zeros(n)
u = ones(n)

P = CQKProblem(d, a, b, r, l, u)

sol, iter, flag = cqk(P)

# Providing an initial guess
x0 = [(rand() .< 0.95) ? 0.0 : 1.0 for i = 1:n]
sol, iter, flag = cqk(P, x0 = x0)

# Pre-allocating the solution vector
sol = similar(P.a)
iter, flag = cqk!(sol, P)
```

Solving a CQK problem on the GPU:
```
using NewtonCQK
using CUDA

T = Float32

n = 10^7
d = CuVector(ones(T, n))
a = CuVector(rand(T, n))
b = CuVector(rand(T, n))
r = T(10.0)
l = CuVector(zeros(T, n))
u = CuVector(ones(T, n))

P = CQKProblem(d, a, b, r, l, u)
sol, iter, flag = cqk(P)
```

Projecting a vector onto a simplex:
```
using NewtonCQK

y = rand(10000)
sol, iter, flag = simplex_proj(y)   # unit simplex
sol, iter, flag = simplex_proj(y, r = 2.0)   # 2-simplex

# Pre-allocating the solution vector
sol = similar(y)
iter, flag = simplex_proj!(sol, y)
```

Projecting a vector onto a simplex in parallel:
```
using NewtonCQK

y = rand(10000)
sol, iter, flag = simplex_proj(y, nchunks=4)   # using 4 threads

# Pre-allocating workspace
sol = similar(y)
chunks = initialize_chunks(10000; nchunks=Threads.nthreads())   # using all available threads
iter, flag = simplex_proj!(sol, y, chunks=chunks)
```

Projecting a vector onto a simplex on the GPU:
```
using NewtonCQK
using CUDA

y = rand(10^7)
cuy = CuVector(y)

# Returning the solution as a new CuVector
sol, iter, flag = simplex_proj(cuy)

# Returning the solution in cuy itself
iter, flag = simplex_proj!(cuy)
```


## Funding

This research was supported by the São Paulo Research Foundation (FAPESP)
(grants 2013/07375-0, 2023/08706-1 and 2024/12967-8) and the National Council
for Scientific and Technological Development (CNPq) (grants 302520/2025-2 and
312394/2023-3), Brazil.


## How to cite

If you use this code in your publications, please cite us, see the
`CITATIONS.bib` in the repository. The 2016 paper is where the Newtow method
for CQK is instroduced. The 2026 arXiv preprint describes the details of this
implementation with numerous benchmarks. 

