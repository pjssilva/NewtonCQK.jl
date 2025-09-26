# Description for all Scripts

## Folder simplex_and_l1ball

See [README.md (in simplex_and_l1ball)](simplex_and_l1ball/README.md) for more details.

## Folder theory_check

See [README.md (in theory_check)](theory_check/README.md) for more details.

## Folder weighted_simplex_and_ball

See [README.md (in weighted_simplex_and_ball)](weighted_simplex_and_ball/README.md) for more details.

## fairness_test.jl

This script uses macro @benchmark to compare runtimes between serial methods (sort and scan, Michelot's method, Condat's method) vs. their corresponding parallel methods with only 1 core separately. We want to use this experiment
to show that we build code for both serial and parallel methods fairly.

Use the following command in your console, then you can get the test result.

```julia
julia> include("fairness_test.jl")
```

## l1ball_runtime_benchmark.jl

This script uses macro @benchmark to test runtimes of 3 serial $\ell_1$ ball projection methods (sort and scan, Michelot's method, Condat's method) and 4 parallel methods (sort and scan, sort and partial scan, Michelot's method, Condat's method) in different numbers of cores (8, 16, 24, 32, 40, 48, 56, 64, 72, 80). The input vector is i.i.d. $N(0,1)$ with size of $10^8$

Use the following command in your console, then you can get the test result.

```julia
julia> include("l1ball_runtime_benchmark.jl")
julia> res_sortscan, res_sortPscan, res_michelot, res_condat = get_result()
julia> res_sortscan #this list includes [serial sort and scan's runtime, parallel sort and scan's runtime in 8, 16, 24, 32, 40, 48, 56, 64, 72, 80 cores]
```

## parity_polytope.jl and paritypolytope_runtime_benckmark.jl

parity_polytope.jl builds two functions for serial and parallel parity polytope projection methods separately. There is a function argument to determine which simplex projection method to use in parity polytope projection.

```julia
julia> include("parity_polytope.jl")
julia> using Random, Distributions
julia> data = rand(1_000_000 - 1). + 1 #generate an input vector i.i.d. U[1, 2] with size of 999999
julia> parity_s(sort_scan_s, data) #using serial sort and scan to finish the projection onto a parity polytope
julia> spa = 0.0001 #set a sparsity rate to terminate parallel computing early
julia> parity_s(michelot_p, data, spa) #using parallel Michelot's method to finish the projection onto a parity polytope
```

paritypolytope_runtime_benckmark.jl uses macro @benchmark to test runtimes of 7 methods in both parity polytope projection and simplex projection (simplex projection is a substep of parity polytope projection) in a given number of
cores (and we set it as nthreads(), which is a system argument in julia). The input vector is i.i.d. $U[1,2]$ with size of $10^8 - 1$.

Use the following command in your console, then you can get the test result.

```julia
julia> include("paritypolytope_runtime_benchmark.jl")
julia> res_parity_polytope, res_simplex = get_result()
julia> res_parity_polytope #this list includes runtimes of [serial sort and scan, parallel sort and scan, parallel and partial scan, serial and parallel Michelot's methods, serial and parallel Condat's methods]
```

## real_data_kdd10.jl and real_data_kdd12.jl

The two scripts use marco @benchmark to test runtimes of simplex projection and l1 ball projection with both serial and parallel (in different numbers of cores) in two real-world data (see [README.md (in data folder)](/data/README.md)), which is a crucial step in the Lasso method.

Use the following command in your console, then you can get the test result.

```julia
julia> include("real_data_kdd10.jl")
julia> res_ss_simplex, res_sps_simplex, res_m_simplex, res_c_simplex, res_ss_ball, res_sps_ball, res_m_ball, res_c_ball = get_result()
julia> res_ss_simplex #this list includes runtims of [serial sort and scan, parallel sort and scan in different cores]
```

### support files

Please make sure to move two real-world datasets with the names kdd10.txt and kdd12.txt to the work folder, and run real_data_reader.jl to preprocess them for real_data_kdd10.jl and real_data_kdd12.jl.

```julia
julia> include("real_data_reader.jl")
```

## simplex_benchmark_runtime.jl

This script uses macro @benchmark to test runtimes of 3 serial simplex projection methods (sort and scan, Michelot's method, Condat's method) and 4 parallel methods (sort and scan, sort and partial scan, Michelot's method, Condat's method) in different numbers of cores (8, 16, 24, 32, 40, 48, 56, 64, 72, 80). There are two groups of runtimes that can be obtained from this script: first, fixing the size of the input vector to $10^8$, and testing runtimes for different input distributions, $U[0,1], N(0,1), N(0, 0.001)$ (using the function get_result_length()); second, fixing the input distribution to $N(0,1)$, and testing runtimes for different sizes of the input vector, $10^7, 10^8, 10^9$ (using the function get_result_norm()). 

Use the following command in your console, then you can get the test result.

```julia
julia> include("simplex_runtime_benchmark.jl")
julia> res_uniform_ss, res_standnorm_ss, res_smallnorm_ss, res_uniform_sps, res_standnorm_sps, res_smallnorm_sps, res_uniroms_m, res_standnorm_m, res_smallnorm_m, res_uniform_c, res_standnorm_c, res_smallnorm_c = get_result_length()
julia> res_uniform_ss #this list includes runtime results as [serial sort and scan, parallel sort and scan in different numbers of cores], and the input distribution is $u[0,1]$
```
## unit_test.jl

This script tests whether 7 simplex projection methods return the same projection result. The benchmark includes 100 vectors with input distribution $u[0,1]$. We want to use this experiment to show we implement parallel method correctly.

Use the following command in your console, then you can get the test result.

```julia
julia> include("unit_test.jl")
```

## wl1ball_benchmark_runtime.jl and wsimplex_benchmark_runtime.jl

The two scripts use macro @benchmark to test runtimes of 3 serial weighted $\ell_1$ ball (weighted simplex) projection methods (sort and scan, Michelot's method, Condat's method) and 3 parallel methods (sort and scan, Michelot's method, Condat's method) in different numbers of cores (8, 16, 24, 32, 40, 48, 56, 64, 72, 80). The input vector is i.i.d. $N(0,1)$ with size of $10^8$, and the weight is i.i.d. $U[0,1]$.

Use the following command in your console, then you can get the test result.

```julia
julia> include("wl1ball_runtime_benchmark.jl")
julia> res_sortscan, res_michelot, res_condat = get_result()
julia> res_sortscan #this list includes [serial sort and scan's runtime, parallel sort and scan's runtime in 8, 16, 24, 32, 40, 48, 56, 64, 72, 80 cores]
```

## plot_graphs.py

This script provides some functions to plot graphs in [results](/results). plot_absolute() can generate a comparing graph for speedup for parallel methods with different numbers of cores vs. the fast serial method (usually is Condat's method). plot_relative_simplex() and plot_relative_others() can generate a comparing graph for speedup for parallel methods with different numbers of cores vs. corresponding serial methods (e.g. parallel sort and scan vs. serial sort and scan).

Here is an example, and more details can be found in [README.md (in results)](/results/README.md)

First, generate runtime results from Julia
```julia
julia> include("simplex_runtime_benchmark.jl")
julia> res_uniform_ss, res_standnorm_ss, res_smallnorm_ss, res_uniform_sps, res_standnorm_sps, res_smallnorm_sps, res_uniroms_m, res_standnorm_m, res_smallnorm_m, res_uniform_c, res_standnorm_c, res_smallnorm_c = get_result_length()
julia> simplex_unif_absolute = [res_uniform_ss[2:end]./res_uniform_c[1], res_uniform_sps[2:end]./res_uniform_c[1], res_uniform_m[2:end]./res_uniform_c[1], res_uniform_c[2:end]./res_uniform_c[1]]
```
Then, you can write simplex_unif_absolute to the file or directly copy it to Python

```python
>>> exec(open("plot_graphs.py").read())
>>> plot_absolute(simplex_unif_absolute, "simplex_unif_comp.png")
```

Then you get
![simplex_unif_comp.png](/results/simplex_unif_comp.png)
