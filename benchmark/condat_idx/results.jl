# Tables and summary of the single-thread comparison produced by runtests.jl.
# Speedups are medians over the instances of the per-instance ratio between the
# time of Condat's original code and the time of each method.
#
#   julia --project condat_idx/results.jl [results_condat_idx.csv]
using CSV, DataFrames, Statistics, Printf

projectpath = isfile("Project.toml") ? "./" : "../"
infile = isempty(ARGS) ? joinpath(projectpath, "results", "results_condat_idx.csv") : ARGS[1]
output_path = joinpath(projectpath, "results", "output")
isdir(output_path) || mkdir(output_path)

w = unstack(CSV.read(infile, DataFrame), [:instance, :n, :rep], :method, :time_ns)
instances = ["Random 1", "Random 2", "Random 3"]
sizes = sort(unique(w.n))
rows(inst, n) = w[(w.instance .== inst) .& (w.n .== n), :]
speedup(inst, n, m) = (r = rows(inst, n); median(r[!, "Condat"] ./ r[!, m]))
# Fraction of the time saved by NewtonCQK.jl that the index-based variant also
# saves, computed from the median times.
function share(inst, n, idx, newton)
    r = rows(inst, n)
    c = median(r[!, "Condat"])
    return (c - median(r[!, idx])) / (c - median(r[!, newton]))
end

outputs = [("dense", "Condat idx", "Newton dense"),
           ("sparse", "Condat idx sparse", "Newton sparse")]

outfile = joinpath(output_path, "table_condat_idx.tex")
open(outfile, "w") do tex
    for (out, idx, newton) in outputs
        println(tex, "% Speedups with respect to Condat's original code, $out solutions (1 thread)")
        println(tex, raw"\begin{tabular}{@{\extracolsep\fill}rrrrrrr@{\extracolsep\fill}}")
        println(tex, raw"    \hline")
        println(tex, "     & " * join([raw"\multicolumn{2}{c}{" * i * "}" for i in instances], " & ") * "\\\\")
        println(tex, raw"    $n$ & " * join(fill(raw"Modified & \texttt{NewtonCQK.jl}", 3), " & ") * "\\\\")
        println(tex, raw"    \hline")
        for n in sizes
            cells = [@sprintf("%.2f", speedup(i, n, m)) for i in instances for m in (idx, newton)]
            println(tex, "    \$10^{$(round(Int, log10(n)))}\$ & " * join(cells, " & ") * " \\\\")
        end
        println(tex, raw"    \hline")
        println(tex, raw"\end{tabular}")
        println(tex)
    end
end
println("Tables written to $outfile")

# Ranges quoted in the text, for n >= 10^7
large = filter(>=(10^7), sizes)
for (out, idx, newton) in outputs
    s_idx = [speedup(i, n, idx) for i in instances for n in large]
    s_new = [speedup(i, n, newton) for i in instances for n in large]
    sh = [share(i, n, idx, newton) for i in instances for n in large]
    @printf("%-6s n >= 1e7: index-based Condat %.2f-%.2f, NewtonCQK.jl %.2f-%.2f, share of time saved %.2f-%.2f\n",
            out, extrema(s_idx)..., extrema(s_new)..., extrema(sh)...)
end
