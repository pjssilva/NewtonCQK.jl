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
