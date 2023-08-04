using RecipesBase, Measures
export InvMcmcSampler, draw_sample, draw_sample!

"""
    InvMcmcSampler(move::InvMcmcMove; kwargs...)

iMCMC sampler for SIS and SIM models using given iMCMC move.
"""
struct InvMcmcSampler{T<:InvMcmcMove}
    move::T
    desired_samples::Int
    burn_in::Int
    lag::Int
    K::Int
    init::McmcInitialiser
    pointers::InteractionSequence{Int}
    function InvMcmcSampler(
        move::T;
        desired_samples::Int=1000, burn_in::Int=0, lag::Int=1,
        K::Int=100, init=InitMode()
    ) where {T<:InvMcmcMove}
        pointers = [Int[] for i in 1:(2K)]
        new{T}(move, desired_samples, burn_in, lag, K, init, pointers)
    end
end

Base.show(io::IO, x::InvMcmcSampler{T}) where {T} = print(io, typeof(x))


acceptance_prob(mcmc::InvMcmcSampler) = acceptance_prob(mcmc.move)

function eval_accept_prob(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    model::SIS,
    log_ratio::Float64
)

    mode, γ, dist = (model.mode, model.γ, model.dist)

    log_lik_ratio = -γ * (
        dist(mode, S_prop) - dist(mode, S_curr)
    )

    return log_lik_ratio + log_ratio

end

function eval_accept_prob(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    model::SIM,
    log_ratio::Float64
)

    mode, γ, dist = (model.mode, model.γ, model.dist)

    log_lik_ratio = -γ * (
        dist(mode, S_prop) - dist(mode, S_curr)
    )

    # For the SIM model we also need to evaluate a multinomial ratio.
    log_multinom_term = log_multinomial_ratio(S_curr, S_prop)
    return log_lik_ratio + log_ratio + log_multinom_term


end

function accept_reject!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    pointers::InteractionSequence{Int},
    move::InvMcmcMove,
    model::T
) where {T<:Union{SIS,SIM}}

    move.counts[2] += 1

    log_ratio = prop_sample!(S_curr, S_prop, move, pointers, model.V)

    # Catch out of bounds proposals (reject them, i.e. 0 acc prob)
    log_α = if any(!(model.K_inner.l ≤ length(x) ≤ model.K_inner.u) for x in S_prop)
        -Inf
    elseif !(model.K_outer.l ≤ length(S_prop) ≤ model.K_outer.u)
        -Inf
    else
        eval_accept_prob(S_curr, S_prop, model, log_ratio)
    end

    # @show log_α
    if log(rand()) < log_α
        # We accept!
        move.counts[1] += 1
        enact_accept!(S_curr, S_prop, pointers, move)
    else
        # We reject!
        enact_reject!(S_curr, S_prop, pointers, move)
    end

end

function intialise_states!(
    pointers::InteractionSequence{Int},
    init::InteractionSequence{Int}
)

    S_curr = InteractionSequence{Int}()
    S_prop = InteractionSequence{Int}()
    for init_path in init
        tmp = pop!(pointers)
        copy!(tmp, init_path)
        push!(S_curr, tmp)
        tmp = pop!(pointers)
        copy!(tmp, init_path)
        push!(S_prop, tmp)
    end

    return S_curr, S_prop

end

function return_states!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    pointers::InteractionSequence{Int}
)
    for i in eachindex(S_curr)
        tmp = pop!(S_curr)
        pushfirst!(pointers, tmp)
        tmp = pop!(S_prop)
        pushfirst!(pointers, tmp)
    end
end

function draw_sample!(
    sample_out::Union{InteractionSequenceSample{Int},SubArray},
    mcmc::InvMcmcSampler,
    model::T;
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
) where {T<:Union{SIS,SIM}}

    pointers = mcmc.pointers

    S_curr, S_prop = intialise_states!(pointers, init)

    sample_count = 1    # Keeps which sample to be stored we are working to get 
    i = 0               # Keeps track all samples (included lags and burn_ins) 
    reset_counts!(mcmc.move) # Reset counts for mcmc move (tracks acceptances)

    while sample_count ≤ length(sample_out)
        i += 1
        # Store value 
        if (i > burn_in) & (((i - 1) % lag) == 0)
            @inbounds sample_out[sample_count] = deepcopy(S_curr)
            sample_count += 1
        end
        accept_reject!(
            S_curr, S_prop,
            pointers,
            mcmc.move,
            model
        )
        # println(S_prop)
    end
    return_states!(S_curr, S_prop, pointers)
end

function draw_sample(
    mcmc::InvMcmcSampler,
    model::T;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
) where {T<:Union{SIS,SIM}}

    sample_out = InteractionSequenceSample{Int}(undef, desired_samples)
    draw_sample!(sample_out, mcmc, model, burn_in=burn_in, lag=lag, init=init)
    return sample_out
end

function (mcmc::InvMcmcSampler)(
    model::T;
    desired_samples::Int=mcmc.desired_samples,
    burn_in::Int=mcmc.burn_in,
    lag::Int=mcmc.lag,
    init::InteractionSequence{Int}=get_init(mcmc.init, model)
) where {T<:Union{SIS,SIM}}

    sample_out = draw_sample(mcmc, model, desired_samples=desired_samples, burn_in=burn_in, lag=lag, init=init)

    return McmcOutput(sample_out, model)

end