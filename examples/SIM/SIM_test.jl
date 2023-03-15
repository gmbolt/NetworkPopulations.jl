using InteractionNetworkModels, Distributions, BenchmarkTools, Plots
using Distances, StructuredDistances
# The Model(s)
model_mode = Hollywood(-3.0, Poisson(7), 10)
S = sample(model_mode, 10)
S = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3, 3, 3]]
V = 1:20

d_lcs = MatchingDistance(FastLCS(100))
d_lcs_t = ThreadedMatchingDistance(LCS())
d_lsp = MatchingDistance(FastLSP(100))
# d_f = FastMatchingDist(FastLCS(100), 51)
d_lsp_fp = FpMatchingDist(FastLSP(100), 100.0)

d_c = CouplingDistance(FastLSP(100))

K_inner = DimensionRange(1, 50)
K_outer = DimensionRange(1, 50)

d_lsp_as = AvgSizeMatchingDist(FastLSP(100), 0.5)
d_lsp_sc = SizeConstrainedMatchingDist(FastLSP(100), 1.0, 4)

model = SIM(S, 5.0, d_lcs, V, K_inner, K_outer)
model_t = SIM(S, 5.0, d_lcs_t, V, K_inner, K_outer)


@time d_lsp_as([[1, 2]], [[1, 2], [1, 3, 3, 4, 5]])
@time d_lsp_sc([[1, 2]], [[1, 2], [1, 3, 3, 4, 5, 12, 3]])
@time d_lsp([[1, 2]], [[1, 2], [1, 3, 3, 4, 5, 12, 3]])

# model_f = SIM(S, 4.0, d_f, V, 50, 50)
mcmc_sampler = SimMcmcInsertDelete(
    ν_ed=5, β=0.6, ν_td=3,
    len_dist=TrGeometric(0.8, K_inner.l, K_inner.u),
    lag=1,
    K=200,
    burn_in=1000
)
mcmc_sampler_sp = SimMcmcInsertDeleteSubpath(
    ν_ed=5, β=0.6, ν_td=3,
    len_dist=TrGeometric(0.1, model.K_inner.l, model.K_inner.u),
    lag=1,
    K=200
)

mcmc_sampler_prop = SimMcmcInsertDeleteProportional(
    ν_ed=1, β=0.6, ν_td=3,
    len_dist=TrGeometric(0.1, model.K_inner.l, model.K_inner.u),
    lag=1,
    K=200, burn_in=1000
)

mcmc_sampler_len = SimMcmcInsertDeleteLengthCentered(
    ν_ed=1, β=0.6, ν_td=3,
    lag=1,
    K=200, burn_in=1000
)

@time out = mcmc_sampler(
    model,
    lag=20,
    init=model.mode,
    burn_in=0,
    desired_samples=200
)


@time out = mcmc_sampler(
    model_t,
    lag=20,
    init=model.mode,
    burn_in=0,
    desired_samples=200
)

plot(out)
summaryplot(out)
out.sample[100]

d_lsp_sc(a, model.mode)

d_lsp_sc([[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2], [3, 4, 4, 4]])


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


