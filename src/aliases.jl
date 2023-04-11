export Path, InteractionSequence, InteractionSequenceSample, InterSeq, InterSeqSample

const Path{T} = Vector{T} where {T<:Union{Int,String}}
const InteractionSequence{T} = Vector{Vector{T}} where {T<:Union{Int,String}}
const InterSeq{T} = Vector{Vector{T}} where {T<:Union{Int,String}}
const InteractionSequenceSample{T} = Vector{Vector{Vector{T}}} where {T<:Union{Int,String}}
const InterSeqSample{T} = Vector{Vector{Vector{T}}} where {T<:Union{Int,String}}
