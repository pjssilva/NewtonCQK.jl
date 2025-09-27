# Convert problems from CPU to GPU and vice-versa
function CPUtoGPU(P::CQKProblem, T::DataType)
    return CQKProblem(
        CuVector(T.(P.d)),
        CuVector(T.(P.a)),
        CuVector(T.(P.b)),
        T(P.r),
        CuVector(T.(P.l)),
        CuVector(T.(P.u))
    )
end
function CPUtoGPU(P::Vector, T::DataType)
    return CuVector(T.(P))
end

function GPUtoCPU(P::CQKProblem, T::DataType)
    return CQKProblem(
        Vector(T.(P.d)),
        Vector(T.(P.a)),
        Vector(T.(P.b)),
        T(P.r),
        Vector(T.(P.l)),
        Vector(T.(P.u))
    )
end
function GPUtoCPU(P::CuVector, T::DataType)
    return Vector(T.(P))
end

# Convert problems to Float32
function F64toF32(P::CQKProblem)
    T = Float32
    return CQKProblem(T.(P.d), T.(P.a), T.(P.b), T(P.r), T.(P.l), T.(P.u))
end
function F64toF32(P::Vector)
    return Float32.(P)
end
