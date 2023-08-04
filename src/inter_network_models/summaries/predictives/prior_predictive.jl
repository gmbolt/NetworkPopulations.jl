using Distributions
export PriorPredictive

"""
Wrapper for predictive distribution implied by given priors for the mode 
and dispersion parameters. 
"""
struct PriorPredictive{T<:Union{SIS,SIM}}
    S_prior::T
    γ_prior::ContinuousUnivariateDistribution
    function PriorPredictive(
        S_prior::T,
        γ_prior::ContinuousUnivariateDistribution
    ) where {T<:Union{SIS,SIM}}
        new{T}(S_prior, γ_prior)

    end
end


"""
    draw_sample(mcmc::InvMcmcSampler, predictive::PosteriorPredictive; kwargs...)

Draw samples from the posterior predictive using the given iMCMC sampler. This involves two steps 
1. Draw parameters from posterior (mode and γ) 
2. Draw sample from model at these parameter values

This has also two keyword arguments 
* `n_samples` = number of draws from the posterior (mode and γ combinations)
* `n_reps` = number of samples from model at each combination of mode and γ
"""
function draw_sample(
    mcmc::InvMcmcSampler,
    predictive::PriorPredictive;
    n_samples::Int=500,  # Number of draws from the posterior 
    n_reps::Int=100  # Number of draws from predictive at sampled parameters 
)

    if n_reps == 1
        out = InteractionSequenceSample{Int}(undef, n_samples)
    else
        out = [InteractionSequenceSample{Int}(undef, n_reps) for i in 1:n_samples]
    end

    acc_prob_out = draw_sample!(out, mcmc, predictive)

    if n_reps == 1
        return out, acc_prob_out
    else
        return vcat(out...), acc_prob_out
    end
end



function draw_sample!(
    out::InteractionSequenceSample{Int},
    mcmc::InvMcmcSampler,
    predictive::PriorPredictive{T};
    mcmc_prior::InvMcmcSampler=mcmc
) where {T<:SIS}
    # Get some model constants 
    S_prior, γ_prior = (
        predictive.S_prior,
        predictive.γ_prior
    )
    # Get number of samples
    n_samples = length(out)
    # Sample from the mode prior 
    S_sample = draw_sample(
        mcmc_prior,
        S_prior,
        desired_samples=n_samples
    )
    # Sample from the dispersion prior
    γ_sample = rand(γ_prior, n_samples)

    # Storage for acceptance probabilities 
    acc_prob_out = zeros(n_samples)

    for i in eachindex(out)
        model = similar(S_prior, S_sample[i], γ_sample[i])
        draw_sample!(view(out, i:i), mcmc, model)
        acc_prob_out[i] = mean(acceptance_prob(mcmc))
    end
    return acc_prob_out
end

function draw_sample!(
    out::Vector{InteractionSequenceSample{Int}}, # Note here we have vector of samples
    mcmc::InvMcmcSampler,
    predictive::PriorPredictive{T};
    mcmc_prior::InvMcmcSampler=mcmc
) where {T<:SIS}

    # Get some model constants 
    S_prior, γ_prior = (
        predictive.S_prior,
        predictive.γ_prior
    )
    # Get number of samples
    n_samples = length(out)

    # Sample from the mode prior 
    S_sample = draw_sample(
        mcmc_prior,
        S_prior,
        desired_samples=n_samples
    )
    # Sample from the dispersion prior
    γ_sample = rand(γ_prior, n_samples)

    # Storage for acceptance probabilities 
    acc_prob_out = zeros(n_samples)

    # For each length n_rep vector in out we 
    # (i) sample parameters from prior (pre-sampled above)
    # (ii) store n_reps samples from model at these parameters
    for i in eachindex(out)
        model = similar(S_prior, S_sample[i], γ_sample[i])
        draw_sample!(out[i], mcmc, model)
        acc_prob_out[i] = mean(acceptance_prob(mcmc))
    end
    return acc_prob_out
end
