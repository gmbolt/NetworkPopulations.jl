using StatsBase
export PosteriorPredictive
export dist_to_mode_sample, mean_inner_dim_sample, outer_dim_sample

"""
A wrapper for the posterior predictive. 

Note: the field `model_type` is used for constructing models.
"""
struct PosteriorPredictive{T<:PosteriorMcmcOutput}
    posterior_out::T
    model_tyle::DataType
    function PosteriorPredictive(posterior_out::T) where {T<:PosteriorMcmcOutput}
        new{T}(posterior, typeof(posterior_out.posterior.S_prior))
    end
end



function draw_sample!(
    out::Vector{Vector{Vector{Int}}},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)

    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(view(out, i:i), mcmc, model)
    end
end

function draw_sample!(
    out::Vector{InteractionSequenceSample{Int}}, # Note difference here with prev function 
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    # Aliases for info in given objects 
    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)

    # For each length n vector in out we 
    # (i) sample parameters from posterior 
    # (ii) store n samples from model at parameters in said vector  
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(out[i], mcmc, model)
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
    predictive::PosteriorPredictive;
    n_samples::Int=500,  # Number of draws from the posterior 
    n_reps::Int=100  # Number of draws from predictive at sampled parameters 
)

    if n_reps == 1
        out = InteractionSequenceSample{Int}(undef, n_samples)
    else
        out = [InteractionSequenceSample{Int}(undef, n_samples) for i in 1:n_reps]
    end

    draw_sample!(out, mcmc, predictive)

    return out
end

"""
    draw_sample_predictive(mcmc::InvMcmcSampler, posterior_out::PosteriorMcmcOutput; kwargs...)

Draw samples from the posterior predictive using the given iMCMC sampler. This involves two steps 
1. Draw parameters from posterior (mode and γ) 
2. Draw sample from model at these parameter values

This has also two keyword arguments 
* `n_samples` = number of draws from the posterior (mode and γ combinations)
* `n_reps` = number of samples from model at each combination of mode and γ
"""
function draw_sample_predictive(
    mcmc::InvMcmcSampler,
    posterior_out::PosteriorMcmcOutput;
    n_samples::Int=500,  # Number of draws from the posterior 
    n_reps::Int=100  # Number of draws from predictive at sampled parameters 
)

    # Aliases
    S_sample = posterior_out.S_sample
    γ_sample = posterior_out.γ_sample
    posterior = posterior_out.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_post_samples = length(S_sample)

    out = if n_reps == 1
        # Each entry of out is a single sample from the posterior
        out = InteractionSequenceSample{Int}(undef, n_samples)
        for i in eachindex(out)
            ind = rand(1:n_post_samples)
            model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
            draw_sample!(view(out, i:i), mcmc, model)
        end
        out
    else
        # Each entry of out is storage for single chain of samples from a model 
        out = [InteractionSequenceSample{Int}(undef, n_samples) for i in 1:n_reps]
        for i in eachindex(out)
            ind = rand(1:n_samples)
            model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
            draw_sample!(out[i], mcmc, model)
        end
        out
    end

    return out
end

# Distance to mode 
# ----------------


function dist_to_mode_sample!(
    out::Vector{Float64},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]]]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = d(sample_store[1], model.mode)
    end
end

function dist_to_mode_sample!(
    out::Vector{Vector{Float64}},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]] for i in 1:length(out[1])]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = map(x -> d(x, model.mode), sample_store)
    end
end

function dist_to_mode_sample(
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive;
    n_samples::Int=500,  # Number of draws from the posterior 
    n_reps::Int=100  # Number of draws from predictive at sampled parameters 
)

    if n_reps == 1
        out = zeros(n_samples)
    else
        out = [zeros(n_samples) for i in 1:n_reps]
    end

    dist_to_mode_sample!(out, mcmc, predictive)

    return out
end

function mean_inner_dim_sample!(
    out::Vector{Float64},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]]]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = mean(lenth.(sample_store[1]))
    end
end

function outer_dim_sample!(
    out::Vector{Int},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]]]

    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = length(sample_store[1])
    end
end

function mean_inner_dim_sample!(
    out::Vector{Vector{Float64}},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]] for i in 1:length(out[1])]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = map(x -> mean(length.(x)), sample_store)
    end
end

function outer_dim_sample!(
    out::Vector{Vector{Int}},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior
    S_sample = posterior_mcmc.S_sample
    γ_sample = posterior_mcmc.γ_sample
    posterior = posterior_mcmc.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_samples = length(S_sample)
    sample_store = [[Int[]] for i in 1:length(out[1])]
    for i in eachindex(out)
        ind = rand(1:n_samples)
        model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
        draw_sample!(sample_store, mcmc, model)
        out[i] = map(x -> length(x), sample_store)
    end
end

function mean_inner_dim_sample(
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive;
    n_samples::Int=500,  # Number of draws from the posterior 
    n_reps::Int=100  # Number of draws from predictive at sampled parameters 
)

    if n_reps == 1
        out = zeros(n_samples)
    else
        out = [zeros(n_samples) for i in 1:n_reps]
    end

    mean_inner_dim_sample!(out, mcmc, predictive)

    return out
end

function outer_dim_sample(
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive;
    n_samples::Int=500,  # Number of draws from the posterior 
    n_reps::Int=100  # Number of draws from predictive at sampled parameters 
)

    if n_reps == 1
        out = zeros(Int, n_samples)
    else
        out = [zeros(Int, n_samples) for i in 1:n_reps]
    end

    outer_dim_sample!(out, mcmc, predictive)

    return out
end