const libcondat = joinpath(dirname(@__FILE__), "condat_simplexproj.so")

function condat_proj(x)
    y = similar(x)
    n = length(x)
    @ccall libcondat.simplexproj_Condat(
        x::Ptr{Cdouble}, y::Ptr{Cdouble}, n::Csize_t, 1.0::Cdouble
    )::Cvoid
    return y
end

function condat_proj!(sol, x)
    n = length(x)
    @ccall libcondat.simplexproj_Condat(
        x::Ptr{Cdouble}, sol::Ptr{Cdouble}, n::Csize_t, 1.0::Cdouble
    )::Cvoid
    return nothing
end

# Index-based variant of Condat's method (see simplexproj_Condat_idx in
# condat_simplexproj.c): same algorithm, but the free variables are tracked by
# their indices instead of copies of their values.
function condat_idx_proj!(sol, x)
    n = length(x)
    @ccall libcondat.simplexproj_Condat_idx(
        x::Ptr{Cdouble}, sol::Ptr{Cdouble}, n::Cuint, 1.0::Cdouble
    )::Cvoid
    return nothing
end

function condat_idx_proj(x)
    sol = similar(x)
    condat_idx_proj!(sol, x)
    return sol
end

# Sparse output. `work` must have length(x) entries.
function condat_idx_spproj(x, work::Vector{Cuint})
    n = length(x)
    tau = Ref{Cdouble}(0.0)
    cnt = @ccall libcondat.simplexproj_Condat_idx_free(
        x::Ptr{Cdouble}, work::Ptr{Cuint}, n::Cuint, 1.0::Cdouble, tau::Ref{Cdouble}
    )::Cint
    inds = Vector{UInt}(undef, cnt)
    vals = Vector{Cdouble}(undef, cnt)
    @inbounds for k in 1:cnt
        i = work[k] + 1
        inds[k] = i
        vals[k] = x[i] - tau[]
    end
    return sparsevec(inds, vals, n)
end

condat_idx_spproj(x) = condat_idx_spproj(x, Vector{Cuint}(undef, length(x)))
