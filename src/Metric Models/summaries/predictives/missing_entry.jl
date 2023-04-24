using StatsBase
export SingleMissingPredictive
export pred_missing, get_prediction, get_truth, get_pred_accuracy

"""
A predictive distribution for a single missing entry. 
"""
struct SingleMissingPredictive
    S::Vector{Vector{Int}}
    ind::Tuple{Int,Int}
    p::Vector{Float64}
end

function Base.show(io::IO, pred::SingleMissingPredictive)
    title = "Missing Entry Predictive Distribution"
    println(io, title)
    println(io, "-"^length(title))
    println(io, "Observation: $(pred.S)")
    println(io, "Missing entry: $(pred.ind)")
end


"""
    pred_missing(S::InterSeq{Int}, ind::Tuple{Int,Int}, posterior_out::PosteriorMcmcOutput)

Posterior predictive for missing entry. Returns instance of `SingleMissingPredictive` for predicting the entry of `S` indexed by `ind`.
"""
function pred_missing(
    S::Vector{Vector{Int}},
    ind::Tuple{Int,Int},
    posterior_out::PosteriorMcmcOutput
)

    posterior = posterior_out.posterior
    d = posterior.dist
    Sₓ = deepcopy(S)
    # Maps a mode to the V (num vertices) different distances to value S with different value in ind 
    dists_to_vals = Dict{Vector{Vector{Int}},Vector{Real}}()
    V = length(posterior.V)
    # @show V
    dist_vec = zeros(V)
    μ_tmp = zeros(V)
    μ = zeros(V)

    for (mode, γ) in zip(posterior_out.S_sample, posterior_out.γ_sample)
        if mode ∉ keys(dists_to_vals)
            for x in 1:V
                Sₓ[ind[1]][ind[2]] = x
                dist_vec[x] = d(Sₓ, mode)
            end
            dists_to_vals[mode] = dist_vec
        end
        map!(x -> exp(-γ * x), μ_tmp, dists_to_vals[mode])
        μ_tmp /= sum(μ_tmp)
        μ += μ_tmp
    end

    μ /= length(posterior_out.S_sample)

    return SingleMissingPredictive(S, ind, μ)
end


"""
    pred_missing(S::InterSeq{Int}, ind::Tuple{Int,Int}, model::Union{SIS,SIM})

True model-based predictive for missing entry. Returns instance of `SingleMissingPredictive` for predicting the entry of `S` indexed by `ind`.
"""
function pred_missing(
    S::Vector{Vector{Int}},
    ind::Tuple{Int,Int},
    model::Union{SIS,SIM}
)

    d, γ = (model.dist, model.γ)
    μ = zeros(length(model.V))
    Sₓ = deepcopy(S)
    i, j = ind
    for x in model.V
        Sₓ[i][j] = x
        μ[x] = exp(-γ * d(Sₓ, model.mode))
    end
    μ /= sum(μ)
    return SingleMissingPredictive(S, ind, μ)
end



"""
    get_prediction(predictive::SingleMissingPredictive)

Query missing entry predictive for prediticted value. Returns tuple of (pred_val, entropy) containing 
the prediction and entropy of the predictive distribution.

Note: if there is a tie, we sample randomly.
"""
function get_prediction(
    pred::SingleMissingPredictive
)
    max_prob = maximum(pred.p)  # MAP 
    vals = findall(pred.p .== max_prob) # Vertices with max MAP
    pred_val = rand(vals) # Choose randomly from said vertices
    H = entropy(pred.p) # Evaluate entropy 
    return pred_val, H
end

"""
    get_truth(pred::SingleMissingPredictive)

Query true entry for missing entry pred. 

"""
function get_truth(
    pred::SingleMissingPredictive
)
    i, j = pred.ind
    return pred.S[i][j]
end

"""
    was_correct(pred::SingleMissingPredictive)

Test whether prediction was correct. Returns Boolean. 
"""
was_correct(pred::SingleMissingPredictive) = (get_prediction(pred)[1] == get_truth(pred))


"""
    get_pred_accuracy(preditives::Vector{SingleMissingPredictive})

Given a vector of missing entry predictives, evaluate the predictive accuracy observed, that is,
the proportion of times the correct prediction was made. 
"""
function get_pred_accuracy(
    preds::Vector{SingleMissingPredictive}
)
    return sum(was_correct.(preds)) / length(preds)
end

