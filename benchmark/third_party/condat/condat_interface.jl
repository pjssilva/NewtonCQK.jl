function condat_proj(x)
    y = similar(x)
    n = length(x)
    libpath = @ccall joinpath(
        dirname(@__FILE__), "condat_simplexproj.so"
    ).simplexproj_Condat(
        x::Ptr{Cdouble}, y::Ptr{Cdouble}, n::Csize_t, 1.0::Cdouble
    )::Cvoid
    return y
end

function condat_proj!(sol, x)
    n = length(x)
    libpath = @ccall joinpath(
        dirname(@__FILE__), "condat_simplexproj.so"
    ).simplexproj_Condat(
        x::Ptr{Cdouble}, sol::Ptr{Cdouble}, n::Csize_t, 1.0::Cdouble
    )::Cvoid
    return nothing
end
