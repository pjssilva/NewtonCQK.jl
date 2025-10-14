using LinearAlgebra

# Project x onto box
function proj!(x, l::Vector{Float64}, u::Vector{Float64})
    @. x = max(l, min(u, x))
end
function proj!(x, l::Float64, u::Float64)
    clamp!(x, l, u)
end

# SPG main function
function spg(n, f, g!, d!, pg_supnorm;
    l = -Inf,
    u = Inf,
    m = 10,
    eta = 1e-4,
    lmin = 1e-10,
    lmax = 1e+10,
    maxiters = 50000,
    eps = 1e-8,
    x0 = Float64[],
    callback = nothing
)
    # -----------------
    # MEMORY ALLOCATION
    # -----------------
    x = Vector{Float64}(undef, n)   # iterate
    xnew = similar(x)               # new iterate
    xbest = similar(x)              # best iterate found so far
    g = similar(x)                  # gradient
    gnew = similar(x)               # new gradient
    d = similar(x)                  # direction
    s = similar(x)                  # xnew - x
    y = similar(x)                  # g(xnew) - g(x)
    lastf = fill(-Inf, m)           # last m values of f

    # --------------
    # INITIALIZATION
    # --------------
    # Initial guess
    isempty(x0) ? x .= 0.0 : x .= x0
    proj!(x, l, u)

    fx = f(x)
    g!(g, x)
    lastf[1] = fx

    # Initial spectral steplength
    tsmall = max(1e-7 * norm(x, Inf), 1e-10)
    d!(d, x, 1.0, g)
    @. s = tsmall * d
    @. xnew = x + s
    g!(gnew, xnew)
    @. y = gnew - g
    sts = dot(s, s)
    sty = dot(s, y)
    lambda = (sty <= 0.0) ? lmax : clamp(sts/sty, lmin, lmax)

    xbest .= x
    fxbest = fx

    iter = 0
    flag = :max_iter

    # ---------
    # MAIN LOOP
    # ---------
    while (true)
        # Projected gradient supnorm
        gsupn = pg_supnorm(x, g)

        println("SPG iteration: $(iter)     |proj g| = $(gsupn)")

        # Test whether convergence is achieved
        if gsupn <= eps
            flag = :solved
            break
        end

        # Test whether maximum number of iterations is achieved
        if iter >= maxiters
            x .= xbest
            break
        end

        # Compute direction
        d!(d, x, lambda, g)

        # Line search
        fxmax = maximum(lastf)
        fx, lsflag = ls!(xnew, f, x, fxmax, g, d, eta)

        if !lsflag
            flag = :too_small_steplength
            x .= xbest
            break
        end

        # Benchmark
        if !isnothing(callback)
            callback(x, iter)
        end

        lastf[mod(iter + 1, m) + 1] = fx

        # Prepare for the next iteration
        g!(gnew, xnew)
        @. s = xnew - x
        @. y = gnew - g
        sts = dot(s, s)
        sty = dot(s, y)
        lambda = (sty <= 0.0) ? lmax : clamp(sts/sty, lmin, lmax)

        x .= xnew
        g .= gnew

        if fx < fxbest
            xbest .= x
            fbest = fx
        end

        iter += 1
    end

    return x, iter, flag
end

# line search
function ls!(xnew, f, x, fxmax, g, d, eta)
    gtd = dot(g, d)

    # initial steplength
    t = 1.0

    @. xnew = x + d
    fxnew = f(xnew)

    tmin = 1e-16

    flag = true
    while (fxnew > fxmax + t * eta * gtd)
        if t <= tmin
            # t is too small, no progress can be expected
            flag = false
            break
        end

        if t <= 0.1
            t *= 2.0
        else
            # quadratic interpolation
            tquad = -0.5 * ( gtd * (t^2) / (fxnew - fx - t * gtd) )

            # backtracking
            if (tquad < 0.1) || (tquad > 0.9 * t)
                t *= 2.0
            else
                t = tquad
            end
        end

        # new trial
        @. xnew = x + t * d
        fxnew = f(xnew)
    end

    return fxnew, flag
end
