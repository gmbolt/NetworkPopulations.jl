export SIM, SimPosterior, Distances

struct SIM{T<:SemiMetric}
    mode::Vector{Path{Int}} # Mode
    γ::Real # Precision
    dist::T # Distance metric
    V::UnitRange{Int} # Vertex Set
    K_inner::DimensionRange # Maximum interaction sequence size
    K_outer::DimensionRange # Maximum path (interaction) length
end

SIM(
    mode::InteractionSequence{Int},
    γ::Real,
    dist::SemiMetric,
    V::UnitRange
) = SIM(
    mode,
    γ,
    dist, V,
    DimensionRange(1, Inf),
    DimensionRange(1, Inf)
)

SIM(
    mode::InteractionSequence{Int},
    γ::Real,
    dist::SemiMetric,
    V::UnitRange,
    K_inner::Real, K_outer::Real
) = SIM(
    mode,
    γ,
    dist, V,
    DimensionRange(1, K_inner),
    DimensionRange(1, K_outer)
)

function Base.show(
    io::IO, model::SIM
)

    title = "$(typeof(model))"
    n = length(title)
    println(io, title)
    println(io, "-"^n)
    println(io, "Parameters:")
    for par in fieldnames(typeof(model))
        println(io, par, " = $(getfield(model, par))")
    end
end


function Base.similar(
    model::SIM{T},
    mode::Vector{Vector{Int}},
    γ::Float64
) where {T}
    return SIM(mode, γ, model.dist, model.V, model.K_inner, model.K_outer)
end

struct SimPosterior{T<:SemiMetric,S<:ContinuousUnivariateDistribution}
    data::InteractionSequenceSample{Int}
    S_prior::SIM{T}
    γ_prior::S
    dist::T
    V::AbstractArray{Int}
    K_inner::DimensionRange
    K_outer::DimensionRange
    sample_size::Int
    function SimPosterior(
        data::InteractionSequenceSample{Int},
        S_prior::SIM{T},
        γ_prior::S
    ) where {T<:SemiMetric,S<:ContinuousUnivariateDistribution}

        dist = S_prior.dist
        V = S_prior.V
        K_inner = S_prior.K_inner
        K_outer = S_prior.K_outer
        sample_size = length(data)
        new{T,S}(data, S_prior, γ_prior, dist, V, K_inner, K_outer, sample_size)
    end
end
