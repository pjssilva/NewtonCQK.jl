using Random, Distributions, BenchmarkTools
#Plots
#import PyPlot
#pyplot()
"""
    test_filter(y, b)
This function is same to Filter but return how many terms remaining after Filter.

# Arguments
- `y::AbstractVector`: the original vector you want to project
- `b::Real = 1`: the scaling factor of the simplex
```
"""
function test_filter(y::AbstractVector, b::Real=1)::Int
    let
        #imitate Filter algorithm
        l = length(y)
        v = [y[1]]
        p = v[1] - b
        u = []
        for i in 2:l
            if y[i] > p
                p = p + (y[i] - p) / (length(v) + 1)
                if p > y[i] - b
                    push!(v, y[i])
                else
                    append!!(u, v)
                    v = [y[i]]
                    p = y[i] - b
                end
            end
        end
        return length(v)
    end
end

"""
    average_filter()
Calculate average number of remaining terms after Filter for different vector sizes
"""
function average_filter()
    let
        #initialize a vector to store results
        res = zeros(19)
        order = 1

        for i in 1:0.5:10
            #for different sizes
            n = Int(i * 1000000)
            #set random seed
            Random.seed!(1234)
            #s is summation
            s = 0
            #repeat 10 times
            for j in 1:10
                data = rand(n)
                s += test_filter(data, 1)
            end
            res[order] = (s / 10)
            order += 1
        end
        return res
    end
end

#calculate results
res1 = average_filter();
#generate (2.2n)^(2/3)
res = (10 ^ 6):(5 * 10 ^ 5):(10 ^ 7)
res2 = zeros(19)
for i in 1:19
    res2[i] = (2.2 * res[i])^(2 / 3)
end
println(res1)
println(res2)
#=
#plot figure
plot(10^6:5*10^5:10^7,res1, linewidth=1, label="observed average", line = (:blue,1,:scatter), xlabel = "input size n",
       ylabel ="remaining elements",legend=:topleft)
plot!(10^6:5*10^5:10^7,res2,label="estimate", line=(:orange, 1, 1))
savefig("filter_result.png")
println("Note: generate a new figure named filter_result.png")
=#
