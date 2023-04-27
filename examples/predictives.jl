using NetworkPopulations, NetworkDistances, Distributions

mode = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
V = 1:5
γ = 3.5
d = EditDist(LCS())
K, L = (10, 20)
true_model = SIS(mode, γ, d, V, K, L)

# Make some pseudo posterior samples 
m = 100_000
S_sample = [deepcopy(mode) for i in 1:m]
γ_sample = [γ for i in 1:m]

S_test = [[1, 1, 1], [2, 2], [3, 3], [1]]
ind = (1, 1)
pred_post = eval_posterior_predictive(
    S_test,
    ind,
    S_sample,
    γ_sample,
    d, V
)

pred_post.p

pred_true = pred_missing(S_test, ind, true_model)

pred_true.p

sum((pred_post.p - pred_true.p) .^ 2)

sum(pred_post.p)

# Testing evaluation on test data 

function test_mse(
    true_model::SIS,
    S_sample::InteractionSequenceSample{Int},
    γ_sample::Vector{Float64},
    mcmc::InvMcmcSampler;
    ntest::Int=100
)
    data_test = mcmc(true_model, desired_samples=ntest).sample
    total_err = 0.0
    counter = 0
    for S in data_test
        for i in eachindex(S)
            for j in eachindex(S[i])
                post_pred = eval_posterior_predictive(
                    S, (i, j),
                    S_sample,
                    γ_sample,
                    true_model.dist,
                    true_model.V
                )
                true_pred = pred_missing(S, (i, j), true_model)
                total_err += sum((post_pred.p - true_pred.p) .^ 2)
                counter += 1
            end
        end
    end
    # Get mean squared error of posteior predictive and true predictive
    return total_err / counter
end

mcmc_sampler = SisMcmcInsertDelete(
    len_dist=TrGeometric(0.8, 1, K),
    ν_ed=5,
    ν_td=2,
    β=0.6,
    lag=10,
    burn_in=4_000,
    K=L + 1
)

post_model = SIS(
    true_model.mode,
    3.8,
    true_model.dist,
    true_model.V,
    true_model.K_inner,
    true_model.K_outer
)
m = 10_000
S_sample = mcmc_sampler(post_model, desired_samples=m).sample
γ_sample = [γ + rand(Normal(0, 0.2)) for i in 1:m]

test_mse(
    true_model,
    S_sample,
    γ_sample,
    mcmc_sampler,
    ntest=10
)
