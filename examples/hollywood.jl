using NetworkPopulations, Distributions


model = Hollywood(-0.1, truncated(Poisson(3), 1, 10), 10)

sample(model, 10)

mean = [zeros(Int, 5) for i in 1:10]
mean1 = sample!([zeros(Int, 5) for i in 1:10], model)
mean
mean1
rand(truncated(Poisson(3), 1, 10), 10)