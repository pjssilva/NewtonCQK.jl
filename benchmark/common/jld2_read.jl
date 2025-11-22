function jld2_read(objectname, filename; test = true)
    output = nothing
    if isfile(filename) && test
        jld2file = jldopen(filename, "r")
        output = read(jld2file, objectname)
        close(jld2file)
    end
    return output
end
