##################################################
# Structures and functions for memory manipulation
##################################################

abstract type AbstractChunk end

mutable struct FixedChunk <: AbstractChunk
    active::Memory{UInt}    # non-fixed indexes
    start::UInt             # start position in active
    final::UInt             # final position in active
    ystart::UInt            # start position in y
    yfinal::UInt            # final position in y
end

function FixedChunk(s, f)
    return FixedChunk(Memory{UInt}(undef, f - s + 1), 0, 0, UInt(s), UInt(f))
end

mutable struct DynamicChunk <: AbstractChunk
    active::Vector{UInt}    # non-fixed indexes
    start::UInt             # start position in active
    final::UInt             # final position in active
    ystart::UInt            # start position in y
    yfinal::UInt            # final position in y
end

DynamicChunk(s, f) = DynamicChunk(Vector{UInt}[], 0, 0, UInt(s), UInt(f))

"""
Pre-allocates workspace and returns the appropriate structure.
This is useful if you need to execute a function several times in sequence.
"""
function initialize_chunks(C::Type, n; numthreads=Threads.nthreads())
    chunks = Vector{C}(undef, numthreads)
    len, inc = divrem(n, numthreads)
    start = 1
    @inbounds for i in 1:inc
        chunks[i] = C(start, start + len)
        start += len + 1
    end
    @inbounds for i in (inc + 1):numthreads
        chunks[i] = C(start, start + len - 1)
        start += len
    end
    # Return balanced chunks
    return chunks
end

function initialize_chunks(n; numthreads=Threads.nthreads())
    return initialize_chunks(FixedChunk, n; numthreads=numthreads)
end

function compress_chunks(chunks::Vector{C}; threshold=512) where {C<:AbstractChunk}
    # Return if there is nothing to do
    if length(chunks) <= 1
        return chunks
    end

    # Join chunks if they are too small
    @inbounds cchunk = chunks[1]
    currentlen = cchunk.final - cchunk.start + 1
    @inbounds for i in 2:length(chunks)
        ichunk = chunks[i]
        ilen = ichunk.final - ichunk.start + 1
        if currentlen + ilen < threshold && cchunk.final + ilen < length(cchunk.active)
            # Copy chunks[i] to current chunk
            cchunk.active[(cchunk.final + 1):(cchunk.final + ilen)] .= ichunk.active[(ichunk.start):(ichunk.final)]
            cchunk.final = cchunk.final + ilen
            ichunk.final = ichunk.start - 1
        else
            cchunk = ichunk
            currentlen = ilen
        end
    end

    # Filter empty chunks
    return filter(c -> c.final >= c.start, chunks)
end

@inline function addfirst!(chunk::FixedChunk)
    @inbounds chunk.active[1] = chunk.ystart
    return 1
end

@inline function addfirst!(chunk::DynamicChunk)
    chunk.active = UInt[]
    sizehint!(chunk.active, max(1024, (chunk.yfinal - chunk.ystart + 1) ÷ 50))
    push!(chunk.active, chunk.ystart)
    return 1
end

@inline function addnext!(chunk::FixedChunk, k::Int, i::UInt)
    k += 1
    @inbounds chunk.active[k] = i
    return k
end

@inline function addnext!(chunk::DynamicChunk, _::Int, i::UInt)
    push!(chunk.active, i)
    return 1
end

@inline lenactive(chunk::FixedChunk, k::Int) = k
@inline lenactive(chunk::DynamicChunk, k::Int) = length(chunk.active)
