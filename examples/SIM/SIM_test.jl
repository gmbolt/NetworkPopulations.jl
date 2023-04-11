using NetworkPopulations, Distributions, BenchmarkTools, Plots
using Distances, NetworkDistances, ProgressMeter
# The Model(s)
model_mode = Hollywood(-3.0, Poisson(7), 10)
S = sample(model_mode, 10)
S = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3, 3, 3]]
V = 1:20
K_inner = DimensionRange(2, 50)
K_outer = DimensionRange(2, 50)

mean_path_len = 0
pen_par = ParametricPenalty(2, 2.0, interc=2.0)
d_par = MatchingDistance(FastLCS(100), pen_par)
d = MatchingDistance(FastLCS(100))


plot([
    [pen_par(zeros(Int, i)) for i in 1:10],
    [d.penalty(zeros(Int, i)) for i in 1:10]
])

d(S, [[1, 1], [1]])
d_par(S, [[1, 1], [1]])

model = SIM(S, 4.5, d_par, V, K_inner, K_outer)
model_dist = SIM(S, 4.5, d, V, K_inner, K_outer)

plot(
    1:50,
    [pen_par(zeros(Int, i)) for i in 1:50],
    xlabel="Path Length",
    ylabel="Penalty"
)


pen_dist = DistancePenalty(LCS())
plot(
    1:50,
    [pen_dist(zeros(Int, i)) for i in 1:50],
    xlabel="Path Length",
    ylabel="Penalty"
)


x = SimMcmcInsertDelete(β=0.2)

typeof(x)

β1 = 0.7
imcmc_move = InvMcmcMixtureMove(
    (
        EditAllocationMove(ν=6),
        InsertDeleteMove(ν=1, len_dist=Geometric(0.5)),
    ),
    (β1, 1 - β1)
)
mcmc_sampler = InvMcmcSampler(
    imcmc_move,
    burn_in=4000, lag=50
)


@time out = mcmc_sampler(
    model,
    lag=20,
    init=model.mode,
    burn_in=0,
    desired_samples=200
)

out.sample[30:40]
acceptance_prob(mcmc_sampler)

summaryplot(out)


plot(out)
plot!(map(x -> d_pen(x, S), out.sample))

# Simulation 
gammas = 1.0:5.0:40.0
scales = 0.1:0.3:2.0

function run(gammas, scales)


    for (gamma_i, scales_i) in product(gammas, scales)
        d = MatchDist(FastLSP(100), ParametricPenalty(mean_path_len, scales, interc=1))
        model = SIM(S, 25.0, d_par, V, K_inner, K_outer)
    end

end


model = SIM(S, 4.8, d_par, V, K_inner, K_outer)


β1 = 0.6
imcmc_move = InvMcmcMixtureMove(
    (
        EditAllocationMove(ν=2),
        InsertDeleteCenteredMove(ν=1)
    ),
    (β1, 1 - β1)
)
mcmc_sampler_centered = InvMcmcSampler(
    imcmc_move,
    burn_in=4000, lag=50
)


@time out = mcmc_sampler_centered(
    model,
    lag=20,
    init=model.mode,
    burn_in=0,
    desired_samples=200
)
out.sample[100]
acceptance_prob(mcmc_sampler_centered)
summaryplot(out)


@time out = mcmc_sampler_len(
    model,
    lag=1,
    init=model.mode,
    burn_in=0,
    desired_samples=10000
)
plot(out)
summaryplot(out)
out.sample


mcmc_sampler_len = SimMcmcInsertDeleteLengthCentered(
    ν_ed=1, β=0.6, ν_td=3,
    lag=1,
    K=200, burn_in=1000
)


mcmc_sampler_len(model)


@time out_prop = mcmc_sampler_prop(
    model,
    lag=1,
    init=model.mode,
    desired_samples=10000
)
plot(out_prop)
summaryplot(out_prop)
out_prop.sample

sample_frechet_var(out.sample, d_lcs, with_memory=true)

n = 1000
m = 4
samples = [draw_sample(mcmc_sampler, model, desired_samples=n) for i in 1:m]

mean_dists_summary(samples, d_lcs)


@time out_sp = mcmc_sampler_sp(
    model,
    lag=1,
    init=model.mode,
    desired_samples=10000,
    burn_in=0
)
plot!(out_sp)

summaryplot(out)
out_sp.sample[1:100]

@btime out = mcmc_sampler(
    model_f,
    lag=1,
    init=model.mode,
    desired_samples=2000,
    burn_in=0
)

plot(out)
summaryplot(out)
S
out.sample

tmp = [1, 2, 1, 2, 1, 4]
V = 1:10

p = zeros(length(V))
p .+= [i ∈ tmp for i in V]
p ./= sum(p)


