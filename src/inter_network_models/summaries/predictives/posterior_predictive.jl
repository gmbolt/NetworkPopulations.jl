using StatsBase, ProgressMeter
export PosteriorPredictive
export dist_to_mode_sample, mean_inner_dim_sample, outer_dim_sample
export draw_sample_predictive

"""
Posterior predictive, functioning as a wrapper around an `PosteriorMcmcOutput`.

Construction: 

        PosteriorPredictive(posterior_out::PosteriorMcmcOutput{T}) 
"""
struct PosteriorPredictive{T<:Union{SisPosterior,SimPosterior}}
    posterior_out::PosteriorMcmcOutput{T}
    model_type::DataType
    function PosteriorPredictive(posterior_out::PosteriorMcmcOutput{T}) where {T<:Union{SisPosterior,SimPosterior}}
        new{T}(posterior_out, typeof(posterior_out.posterior.S_prior))
    end
end

"""
    draw_sample!(out::InterSeqSample{Int}, mcmc::InvMcmcSampler, predictive::PosteriorPredictive)

Draw samples from posterior predictive via MCMC sampler and store in `out` in-place. This will draw a *single* sample 
from the model for each sample from the posterior. 
"""
function draw_sample!(
    out::InterSeqSample{Int},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior_out
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

"""
    draw_sample!(out::Vector{InterSeqSample{Int}}, mcmc::InvMcmcSampler, predictive::PosteriorPredictive)

Draw samples from posterior predictive via MCMC sampler and store in `out` in-place. This will draw multiple samples
from the model for each sample from the posterior. 
"""
function draw_sample!(
    out::Vector{InterSeqSample{Int}}, # Note difference here with prev function 
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    # Aliases for info in given objects 
    posterior_mcmc = predictive.posterior_out
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
        out = InterSeqSample{Int}(undef, n_samples)
    else
        out = [InterSeqSample{Int}(undef, n_samples) for i in 1:n_reps]
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
    n_reps::Int=100,  # Number of draws from predictive at sampled parameters 
    loading_bar::Bool=false # Whether to print a loading bar
)

    # If printing loading bar, get progress meter 
    if loading_bar
        iter = Progress(
            n_samples, # How many iters 
            "Sampling from posterior predictive....")  # Loading bar. Minimum update interval: 1 second
    end
    
    # Aliases
    S_sample = posterior_out.S_sample
    γ_sample = posterior_out.γ_sample
    posterior = posterior_out.posterior
    d = posterior.dist
    V = posterior.V
    K_I, K_O = (posterior.K_inner, posterior.K_outer)
    n_post_samples = length(S_sample)

    # Instantiate predictive 
    predictive = PosteriorPredictive(posterior_out)

    out = if n_reps == 1
        # Each entry of out is a single sample from the posterior
        out = InterSeqSample{Int}(undef, n_samples)
        for i in eachindex(out)
            ind = rand(1:n_post_samples)
            model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
            draw_sample!(view(out, i:i), mcmc, model)
            if loading_bar
                next!(iter)
            end
        end

        out
    else
        # Each entry of out is storage for single chain of samples from a model 
        out = [InterSeqSample{Int}(undef, n_samples) for i in 1:n_reps]
        for i in eachindex(out)
            ind = rand(1:n_samples)
            model = predictive.model_type(S_sample[ind], γ_sample[ind], d, V, K_I, K_O)
            draw_sample!(out[i], mcmc, model)
            if loading_bar
                next!(iter)
            end
        end
        out
    end

    return out
end

"""
    dist_to_mode_sample!(out::Vector{Float64}, mcmc::InvMcmcSampler, predictive::PosteriorPredictive)

Draw sample from posterior predictive distribution of distances to the mode and store in `out` in-place, with a single sample from model
for each sample from posterior. 
"""
function dist_to_mode_sample!(
    out::Vector{Float64},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior_out
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

"""
    dist_to_mode_sample!(out::Vector{Vector{Float64}}, mcmc::InvMcmcSampler, predictive::PosteriorPredictive)

Draw sample from posterior predictive distribution of distances to the mode and store in `out` in-place, with mutiple samples from model
for each sample from posterior.
"""
function dist_to_mode_sample!(
    out::Vector{Vector{Float64}},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior_out
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

"""
    dist_to_mode_sample(mcmc::InvMcmcSampler, predictive::PosteriorPredictive; kwargs...)

Draw sample from posterior predictive distribution of distances to the mode. This involves two steps 
1. Draw parameters from posterior (mode and γ) 
2. Draw sample from model at these parameter values and store distance to the mode for each sample 
    
This has also two keyword arguments 
* `n_samples` = number of draws from the posterior (mode and γ combinations)
* `n_reps` = number of samples from model at each combination of mode and γ
"""
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


"""
    mean_inner_dim_sample!(out::Vector{Float64}, mcmc::InvMcmcSampler, predictive::PosteriorPredictive)

Draw sample from posterior predictive distribution of mean inner dimension (path length) and store in `out` in-place, with a single sample from model
for each sample from posterior. 
"""
function mean_inner_dim_sample!(
    out::Vector{Float64},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior_out
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

"""
    mean_inner_dim_sample!(out::Vector{Vector{Float64}}, mcmc::InvMcmcSampler, predictive::PosteriorPredictive)

Draw sample from posterior predictive distribution of mean inner dimension (path length) and store in `out` in-place, with mutiple samples from model
for each sample from posterior.
"""
function mean_inner_dim_sample!(
    out::Vector{Vector{Float64}},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior_out
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


"""
    mean_inner_dim_sample(mcmc::InvMcmcSampler, predictive::PosteriorPredictive; kwargs...)

Draw sample from posterior predictive distribution of mean inner dimension (path length). This involves two steps 
1. Draw parameters from posterior (mode and γ) 
2. Draw sample from model at these parameter values and store mean path length for each sample
    
This has also two keyword arguments 
* `n_samples` = number of draws from the posterior (mode and γ combinations)
* `n_reps` = number of samples from model at each combination of mode and γ
"""
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


"""
    outer_dim_sample!(out::Vector{Int}, mcmc::InvMcmcSampler, predictive::PosteriorPredictive)

Draw sample from posterior predictive distribution of outer dimension (number of paths) and store in `out` in-place, with a single sample from model
for each sample from posterior. 
"""
function outer_dim_sample!(
    out::Vector{Int},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior_out
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


"""
    mean_inner_dim_sample!(out::Vector{Vector{Int}}, mcmc::InvMcmcSampler, predictive::PosteriorPredictive)

Draw sample from posterior predictive distribution of outer dimension (number of paths) and store in `out` in-place, with mutiple samples from model
for each sample from posterior.
"""
function outer_dim_sample!(
    out::Vector{Vector{Int}},
    mcmc::InvMcmcSampler,
    predictive::PosteriorPredictive
)

    posterior_mcmc = predictive.posterior_out
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

"""
    outer_dim_sample(mcmc::InvMcmcSampler, predictive::PosteriorPredictive; kwargs...)

Draw sample from posterior predictive distribution of outer dimension (number of paths). This involves two steps 
1. Draw parameters from posterior (mode and γ) 
2. Draw sample from model at these parameter values and store number of paths in each sample
    
This has also two keyword arguments 
* `n_samples` = number of draws from the posterior (mode and γ combinations)
* `n_reps` = number of samples from model at each combination of mode and γ
"""
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