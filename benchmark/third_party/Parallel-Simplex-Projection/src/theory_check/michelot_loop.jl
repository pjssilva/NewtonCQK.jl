using Random, Distributions
#using Plots

"""
    test_michelot(y, b)
This function records numbers of remaining terms after each iteration in michelot
    method. It will return a vector includes all numbers.

# Arguments
- `y::AbstractVector`: the original vector you want to project
- `a::Int = 1`: the scaling factor of the simplex
```
"""
function test_michelot(y::AbstractVector, b::Real=1)::AbstractVector
    let
        #imitate michelot method
        u = []
        #initial pivot
        p = (sum(y) - b) / length(y)
        while true
            l = length(y)
            for i in 1:l
                x = popfirst!(y)
                if x > p
                    push!(y, x)
                end
            end
            p = (sum(y) - b) / length(y)
            if l == length(y)
                return u
            else
                push!(u, (l - length(y)))
            end
        end
    end
end

"""
    average_loop()
Calculate average number of remaining terms after all iterations in Michelot method
"""
function average_loop()
    let
        #initialize a vector to store results
        s = zeros(12)
        #set random seed
        Random.seed!(1234)
        t = 0
        #repeat 100 times
        for i in 1:101
            u = test_michelot(rand(1000000), 1)
            if length(u) == 12
                t += 1
                s = s + u
            end
        end
        s = s ./ (t)
        return s
    end
end

#calculate results
res1 = average_loop();
#generate a geometric series with a ratio of 1/2
res2 = zeros(12)
let
    s = 1000000
    for i in 1:12
        s = s / 2
        res2[i] = s
    end
end
println(res1)
println(res2)
#=
#plot figure
plot(1:12,res1, xticks=1:1:12, linewidth=2, label="observed average", line = (:blue,1,:scatter), xlabel = "iterations",
       ylabel ="remaining elements")
plot!(1:12,res2,label="estimate", line=(:orange, 1, 1))
savefig("michelot_loop.png")
println("Note: generate a new figure named michelot_loop.png")
=#
