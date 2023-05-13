using NetworkPopulations, NetworkDistances, Distributions, StatsBase, DrWatson
using ProgressMeter

mode = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
V = 1:5
γ = 3.5
d = EditDist(LCS())
K, L = (10, 20)
true_model = SIS(mode, γ, d, V, K, L)

# Make some pseudo-posterior samples 
m = 100_000
S_sample = [deepcopy(mode) for i in 1:m]
γ_sample = [γ for i in 1:m]

S_test = [[1, 1, 1], [2, 2], [3, 3], [1]]
ind = (4, 1)
pred_post = eval_posterior_predictive(
    S_test,
    ind,
    S_sample,
    γ_sample,
    d, V
)

get_prediction(pred_post)

pred_post.p

pred_true = pred_missing(S_test, ind, true_model)


length(symdiff(get_prediction(pred_true), get_prediction(pred_post)))


pred_true.p

sum((pred_post.p - pred_true.p) .^ 2)

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
                # total_err += sum((post_pred.p - true_pred.p) .^ 2)
                total_err += kldivergence(true_pred.p, post_pred.p)
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
    3.9,
    true_model.dist,
    true_model.V,
    true_model.K_inner,
    true_model.K_outer
)
m = 10_000
S_sample = mcmc_sampler(post_model, desired_samples=m).sample
γ_sample = [γ - abs(rand(Normal(0, 0.2))) for i in 1:m]


d_mean = mean(d(true_model.mode, S_i) for S_i in S_sample)

test_mse(
    true_model,
    S_sample,
    γ_sample,
    mcmc_sampler,
    ntest=100
)

# Simulation 
mode = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
V = 1:5
γ = 3.5
d = EditDist(LCS())
K, L = (10, 20)
true_model = SIS(mode, γ, d, V, K, L)

function run_sim(config_i::Dict)

    post_sampler = SisMcmcInsertDelete(
        len_dist=TrGeometric(0.8, 1, K),
        ν_ed=5,
        ν_td=2,
        β=0.6,
        lag=30,
        burn_in=4_000,
        K=config_i[:outer_dim] + 1
    )
    data_sampler = SisMcmcInsertDelete(
        len_dist=TrGeometric(0.8, 1, K),
        ν_ed=5,
        ν_td=2,
        β=0.6,
        lag=100,
        burn_in=10_000,
        K=config_i[:outer_dim] + 1
    )
    hw_model = Hollywood(-0.35, truncated(Poisson(3), 1, Inf), config_i[:nvertices])
    mode = sample!([zeros(Int, n_i) for n_i in config_i[:true_mode_dims]], hw_model)
    println(mode)
    true_model = SIS(
        mode,
        config_i[:gamma_true],
        config_i[:distance],
        1:config_i[:nvertices],
        config_i[:inner_dim],
        config_i[:outer_dim]
    )
    post_model = SIS(
        true_model.mode,
        config_i[:post_conc],
        true_model.dist,
        true_model.V,
        true_model.K_inner,
        true_model.K_outer
    )
    m = config_i[:nsamples]
    S_sample = post_sampler(post_model, desired_samples=m).sample
    γ_sample = [γ - abs(rand(Normal(0, config_i[:gamma_noise]))) for i in 1:m]


    d_mean = mean(d(true_model.mode, S_i) for S_i in S_sample)

    err = test_mse(
        true_model,
        S_sample,
        γ_sample,
        data_sampler,
        ntest=config_i[:ntest]
    )

    return (d_mean, err)

end

config = Dict(
    :post_conc => [3.8, 3.6, 3.4, 3.2],
    :ntest => [10, 100],
    :inner_dim => 10,
    :outer_dim => 20,
    :gamma_noise => 0.3,
    :nsamples => 500,
    :true_mode_dims => [[3, 3, 3, 3, 3]],
    :nvertices => 5,
    :distance => EditDist(LCS()),
    :gamma_true => 3.7,
    :index => collect(1:10)
)

config_list = dict_list(config)

sim_out = @showprogress [run_sim(config_i) for config_i in config_list]

scatter(
    sim_out,
    group=[config_i[:ntest] for config_i in config_list]
)