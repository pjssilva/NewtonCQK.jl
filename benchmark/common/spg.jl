# SPG main function
# n: number of variables
# f: objective function -- f(x)
# g!: gradient of f -- g!(g, x) stores grad f in g
# proj!: projection operator -- proj!(p, z, x0) stores P(z) in p
#        we pass x0 = x as warm start for CQK solver
function spg(n, f, g!, proj!;
    l = -Inf,
    u = Inf,
    m = 10,
    eta = 1e-4,
    lmin = 1e-10,
    lmax = 1e+10,
    maxiters = 50000,
    eps = 1e-6,
    x0 = Float64[],
    callback = nothing,
    verbose = 1
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
    # Update x by its projection, with warm start x
    proj!(x, x, x)

    fx = f(x)
    g!(g, x)
    gsupn = Inf
    lastf[1] = fx

    # Initial spectral steplength
    tsmall = max(1e-7 * norm(x, Inf), 1e-10)
    proj!(d, x, g)
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
        # Projected gradient supnorm (use d as auxiliary vector)
        proj!(d, x .- g, x)
        d .-= x
        gsupn = norm(d, Inf)

        if verbose > 0
            @printf("SPG iter: %8d    |proj g| = %12.8e    f = %12.8e\n", iter, gsupn, fx)
        end

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

        iter += 1

        # Compute direction
        if !proj!(d, x .- lambda * g, x)
            flag = :error_proj
            break
        end
        d .-= x

        # Callback
        # To compute direction, we project x .- lambda * g starting at x
        if !isnothing(callback)
            if callback(x .- lambda * g, x, iter)
                flag = :callback_stop
                break
            end
        end

        # Line search
        fxmax = maximum(lastf)
        fx, lsflag = ls!(xnew, f, x, fx, fxmax, g, d, eta)

        if !lsflag
            flag = :too_small_steplength
            x .= xbest
            break
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
    end

    println("\nSPG exit status: $(flag)")
    @printf("\nSummary: iters = %d    |proj g| = %12.8e    f = %12.8e\n", iter, gsupn, fx)

    return x, iter, flag
end

# line search
function ls!(xnew, f, x, fx, fxmax, g, d, eta)
    gtd = dot(g, d)

    # initial steplength
    t = 1.0

    @. xnew = x + d
    fxnew = f(xnew)

    tmin = 1e-14

    flag = true
    while (fxnew > fxmax + t * eta * gtd)
        if t <= tmin
            # t is too small, no progress can be expected
            flag = false
            break
        end

        if t <= 0.1
            t /= 2.0
        else
            # quadratic interpolation
            tquad = -0.5 * ( gtd * (t^2) / (fxnew - fx - t * gtd) )

            # backtracking
            if (tquad < 0.1) || (tquad > 0.9 * t)
                t /= 2.0
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
