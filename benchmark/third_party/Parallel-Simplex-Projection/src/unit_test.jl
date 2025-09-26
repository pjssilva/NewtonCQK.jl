#unit_test.jl

using Random, Distributions

include("simplex_and_l1ball/simplex_wrap.jl")

println("You are using ", nthreads(), " threads for parallel computing")

#just check pivots from both serial and parallel methods; thus write new functions
function sortscan_serial_pivot(data::AbstractVector, b::Real=1)::Real
    return sortscan_pivot(data, b)
end

function sortPscan_parallel_pivot(data::AbstractVector, b::Real=1)::Real
    d_after_sort = sort(data; alg=ThreadsX.MergeSort, rev=true)
    return parallel_partial_scan(d_after_sort, b)
end

function sortscan_unit_test(toler::Float64=10^(-7))
    acculumate_error_1 = 0
    acculumate_error_2 = 0
    numer_extoler_1 = 0
    numer_extoler_2 = 0
    for _ in 1:100
        data = rand(10_000_000)
        pivot_serial = sortscan_serial_pivot(data[1:end], 1)
        pivot_parallel_1 = sortscan_parallel_pivot(data[1:end], 1)
        pivot_parallel_2 = sortPscan_parallel_pivot(data[1:end], 1)
        error = abs(pivot_serial - pivot_parallel_1)
        (error > toler) && (numer_extoler_1 += 1)
        acculumate_error_1 += error
        error = abs(pivot_serial - pivot_parallel_2)
        (error > toler) && (numer_extoler_2 += 1)
        acculumate_error_2 += error
    end
    println(
        "Number of instances having error larger than tolerance for sortscan_p: ",
        numer_extoler_1,
    )
    println("Average errors for sortscan_p: ", acculumate_error_1)
    println(
        "Number of instances having error larger than tolerance for sortPscan_p: ",
        numer_extoler_2,
    )
    return println("Average errors for sortPscan_p: ", acculumate_error_2)
end

#just check pivots from both serial and parallel methods; thus write new functions
function michelot_serial_pivot(data::AbstractVector, b::Real=1)::Real
    return serial_scan(data[1:end], b)
end

function michelot_parallel_pivot(
    data::AbstractVector, b::Real=1, numthread::Int=nthreads(), spa::Float64=0.01
)::Real
    active_list = parallel_scan(data, b, numthread, spa)
    return serial_scan(active_list, b)
end

function michelot_unit_test(toler::Float64=10^(-7))
    acculumate_error = 0
    numer_extoler = 0
    for _ in 1:100
        data = rand(10_000_000)
        pivot_serial = michelot_serial_pivot(data, 1)
        pivot_parallel = michelot_parallel_pivot(data, 1)
        error = abs(pivot_serial - pivot_parallel)
        (error > toler) && (numer_extoler += 1)
        acculumate_error += error
    end
    println("Number of instances having error larger than tolerance: ", numer_extoler)
    return println("Average errors: ", acculumate_error)
end

#just check pivots from both serial and parallel methods; thus write new functions
function condat_serial_pivot(data::AbstractVector, b::Real=1)::Real
    active_list = serial_filter(data, b)
    return checkL(active_list, b)
end

function condat_parallel_pivot(
    data::AbstractVector, b::Real=1, numthread::Int=nthreads(), spa::Float64=0.01
)::Real
    active_list = parallel_filter(data, b, numthread, spa)
    return checkL(active_list, b)
end

function condat_unit_test(toler::Float64=10^(-7))
    acculumate_error = 0
    numer_extoler = 0
    for _ in 1:100
        data = rand(10_000_000)
        pivot_serial = condat_serial_pivot(data, 1)
        pivot_parallel = condat_parallel_pivot(data, 1)
        error = abs(pivot_serial - pivot_parallel)
        (error > toler) && (numer_extoler += 1)
        acculumate_error += error
    end
    println("Number of instances having error larger than tolerance: ", numer_extoler)
    return println("Average errors: ", acculumate_error)
end

function sortscan_michelot_unit_test(toler::Float64=10^(-7))
    acculumate_error = 0
    numer_extoler = 0
    for _ in 1:100
        data = rand(10_000_000)
        pivot_sortscan = sortscan_serial_pivot(data, 1)
        pivot_michelot = michelot_serial_pivot(data, 1)
        error = abs(pivot_michelot - pivot_sortscan)
        (error > toler) && (numer_extoler += 1)
        acculumate_error += error
    end
    println("Number of instances having error larger than tolerance: ", numer_extoler)
    return println("Average errors: ", acculumate_error)
end

function sortscan_condat_unit_test(toler::Float64=10^(-7))
    acculumate_error = 0
    numer_extoler = 0
    for _ in 1:100
        data = rand(10_000_000)
        pivot_sortscan = sortscan_serial_pivot(data, 1)
        pivot_condat = condat_serial_pivot(data, 1)
        error = abs(pivot_condat - pivot_sortscan)
        (error > toler) && (numer_extoler += 1)
        acculumate_error += error
    end
    println("Number of instances having error larger than tolerance: ", numer_extoler)
    return println("Average errors: ", acculumate_error)
end

function michelot_condat_unit_test(toler::Float64=10^(-7))
    acculumate_error = 0
    numer_extoler = 0
    for _ in 1:100
        data = rand(10_000_000)
        pivot_michelot = michelot_serial_pivot(data, 1)
        pivot_condat = condat_serial_pivot(data, 1)
        error = abs(pivot_michelot - pivot_condat)
        (error > toler) && (numer_extoler += 1)
        acculumate_error += error
    end
    println("Number of instances having error larger than tolerance: ", numer_extoler)
    return println("Average errors: ", acculumate_error)
end

println("sortscan unit test: ")
Random.seed!(12345);
sortscan_unit_test();
println("michelot unit test: ")
Random.seed!(12345);
michelot_unit_test();
println("codnat unit test: ")
Random.seed!(12345);
condat_unit_test();
println("sortscan vs michelot unit test: ")
Random.seed!(12345);
sortscan_michelot_unit_test();
println("sortscan vs condat unit test: ")
Random.seed!(12345);
sortscan_condat_unit_test();
println("michelot vs condat unit test: ")
Random.seed!(12345);
michelot_condat_unit_test();
