using DataFrames
using JLD2
using OpenML

include("dataset.jl")
include("../common/jld2_read.jl")

# Project path
projectpath = isfile("Project.toml") ? "./" : "../"

datasets_file = joinpath(projectpath, "svm", "datasets.jld2")

# Download selected datasets with OpenML
function try_load_dataset(id)
    datasets = jld2_read("datasets", datasets_file)
    if isnothing(datasets)
        datasets = DATASET[]
    end

    for s in datasets
        if s.id == id
            println("Dataset $(s.name) already downloaded")
            return nothing, nothing, false
        end
    end

    list = []
    try
        list = OpenML.list_datasets(filter="data_id/$(id)", output_format = DataFrame)
    catch
        println("Error while list datasets with OpenML.")
        println("This problem occurs probably due when try to access the server.")
        return nothing, nothing, true
    end

    if isempty(list)
        println("Dataset id not found.")
        return nothing, nothing, false
    end

    d = list[1,:]

    data = DataFrame()
    try
        data = DataFrame(OpenML.load(d.id))
    catch
        println("Fail to load dataset $(d.name), id $(d.id)")
        return nothing, nothing, true
    end

    return d.name, data, false
end

function load_dataset(id)
    trials = 0
    name = ""
    data = DataFrame()
    while true
        trials += 1
        name, data, fail = try_load_dataset(id)
        if fail
            if trials >= 10
                println("Fail to download datasets. Try again!")
                return nothing, nothing
            end
            println("Try again in 10 seconds...")
            sleep(10)
        else
            break
        end
    end
    return name, data
end

# Main function
function main()
    println("=========\nDownloading SVM datasets\n=========")

    datasets = jld2_read("datasets", datasets_file)
    if isnothing(datasets)
        datasets = DATASET[]
    end

    ################
    # MNIST (id 554)
    ################
    id = 554
    name, data = load_dataset(id)
    if !isnothing(data)
        # Index of all training instances with digit 8
        # (last column contains the labels)
        digit8 = sort(findall(data[1:60000,end] .== "8"))

        # Other digits
        otherdigits = sort(findall(data[1:60000,end] .!= "8"))

        # Select instances (all "8", first length(digit8) of other digits)
        selected = sort(union(digit8, otherdigits[1:length(digit8)]))

        # Filter data
        data = data[selected,:]

        # "8" => 1, other => -1
        data[data[:,end] .!= "8", end] .= "-1"
        data[data[:,end] .== "8", end] .= "1"

        # Normalize data
        data[:,1:(end-1)] ./= 255

        push!(datasets, DATASET(id, name, size(data,2)-1, size(data,1), data))

        println("$(name) downloaded")
    end

    ################
    # cdc_diabetes (id 46598)
    ################
    id = 46598
    name, data = load_dataset(id)
    if !isnothing(data)
        N = 10_000

        # Index first N training instances with diabetes
        yes = sort(findall(data[:,end] .== 1.0))

        # No diabetes
        no = sort(findall(data[:,end] .== 0.0))

        # Select instances (all "yes", first length(yes) "no")
        selected = sort(union(yes[1:N], no[1:N]))

        # Filter data
        data = data[selected,:]

        push!(datasets, DATASET(id, name, size(data,2)-1, size(data,1), data))

        println("$(name) downloaded")
    end

    jldsave(datasets_file; datasets)

    return 0
end

# Run main if non-iteractive
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
