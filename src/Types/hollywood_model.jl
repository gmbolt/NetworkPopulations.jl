using Distributions, StatsBase
export Hollywood

struct Hollywood
    α::Real
    θ::Real
    ν::DiscreteUnivariateDistribution
    K::Real
    function Hollywood(α::Real, θ::Real, ν::DiscreteUnivariateDistribution, K::Real)
        # @show θ, -K*α
        @assert ((0 < α < 1.0) & (θ > -α)) | ((α < 0) & (θ == (-K * α))) "Check parameters satisfy required constraints"
        new(α, θ, ν, K)
    end
end

Hollywood(α::Real, θ::Real, ν::DiscreteUnivariateDistribution) = Hollywood(α, θ, ν, Inf)
Hollywood(α::Real, ν::DiscreteUnivariateDistribution, K::Int) = Hollywood(α, (-K) * α, ν, K)

function StatsBase.sample!(
    out::InteractionSequence{Int},
    model::Hollywood
)
    # Init
    t = 0
    V = 0
    counts = Int[]
    prob_new(t, V, θ, α) = (θ + α * V) / (t - 1 + θ)
    for i in eachindex(out)
        # m = rand(model.ν)
        I = out[i]
        for j in eachindex(I)
            t += 1
            if t == 1
                I[j] = 1
                V += 1
                push!(counts, 1)
            elseif rand() < prob_new(t, V, model.θ, model.α)
                I[j] = V + 1
                V += 1
                push!(counts, 1)
            else
                μ = cumsum(counts) / sum(counts)
                pushfirst!(μ, 0.0)
                # @show μ
                val, p = rand_multivariate_bernoulli(μ)
                I[j] = val
                counts[val] += 1
            end
            # out[i] = I_tmp
        end
    end
    return out
end

function StatsBase.sample(
    model::Hollywood,
    n::Int
)
    out = [zeros(Int, rand(model.ν)) for i in 1:n]
    sample!(out, model)
    return out
end
