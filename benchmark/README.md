The script `run_alltests.sh` reproduces the tests in the reference paper.
Adjust the variables `juliacmd` and `threads` to your hardware before running.
The output figures and tables are stored in the folder `results/output`.
Make sure that all packages in the project are installed in your Julia
environment.

Obs: You need to use Julia 1.13 to fully reproduce the results. The committed
`Manifest.toml` pins the package versions that were used; `Pkg.instantiate()`
in this folder installs exactly those. With another Julia version you have to
delete `Manifest.toml` and re-resolve, and since the `[sources]` entry in
`Project.toml` requires Julia 1.11 or newer, on older versions you also need
`Pkg.develop(path="..")` to point at the local `NewtonCQK.jl`.

The folder `condat_idx` measures how much of the single-thread advantage of
`NewtonCQK.jl` over Condat's code comes from tracking the indices of the free
variables instead of their values. The function `simplexproj_Condat_idx`, added
to `third_party/condat/condat_simplexproj.c` next to Condat's original code
(which is unchanged), implements the same method with that data structure, and
`simplexproj_Condat_idx_free` returns the free variables to build a sparse
solution. `condat_idx/check.jl` verifies that both reproduce Condat's output bit
for bit, `condat_idx/runtests.jl` runs the benchmark, writing
`results/results_condat_idx.csv`, and `condat_idx/results.jl` writes the tables
to `results/output/table_condat_idx.tex`. Run the benchmark with one thread; the
results in the paper were obtained on a single pinned core (`taskset -c 2`).
The variant of Condat's code was developed with the assistance of Anthropic's
Claude Opus 5.5.
