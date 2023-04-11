using RecipesBase
export McmcOutput, PosteriorMcmcOutput

# TODO - add any used summaries to this file 

struct McmcOutput{T<:Union{SIS,SIM}}
    sample::Vector{Vector{Path{Int}}}  # The sample
    model::T
end

Base.show(io::IO, x::McmcOutput) = print(io, typeof(x))

@recipe function f(output::McmcOutput)
    model = output.model
    sample = output.sample
    x = map(x -> model.dist(model.mode, x), sample)
    xguide --> "Sample"
    yguide --> "Distance from Mode"
    legend --> false
    size --> (800, 300)
    margin --> 5mm
    x
end

struct PosteriorMcmcOutput{T<:Union{SisPosterior,SimPosterior}}
    S_sample::Vector{InteractionSequence{Int}}
    γ_sample::Vector{Float64}
    suff_stat_trace::Vector{Float64}
    posterior::T
end

Base.show(io::IO, x::PosteriorMcmcOutput) = print(io, typeof(x))


@recipe function f(
    output::PosteriorMcmcOutput,
    S_true::InteractionSequence{Int}
)

    S_sample = output.S_sample
    γ_sample = output.γ_sample
    layout := (2, 1)
    legend --> false
    xguide --> "Index"
    yguide --> ["Distance from True Mode" "γ"]
    size --> (800, 600)
    margin --> 5mm
    y1 = map(x -> output.posterior.dist(S_true, x), S_sample)
    y2 = γ_sample
    hcat(y1, y2)
end