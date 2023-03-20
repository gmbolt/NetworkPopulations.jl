# Here we just check model samplers work with all moves 
@testset "imcmc_moves" begin

    # Fixed parameters 
    V = 1:4
    n = 3
    gamma = 4.5
    mode = [rand(V, rand(4:7)) for i in 1:n]
    K_in, K_out = (10, 10)
    d_ground = FastLCS(11)
    d_sis = FastEditDistance(d_ground, 11)
    d_sim = MatchingDistance(d_ground)

    # Fixed models 
    model_sis = SIS(mode, gamma, d_sis, V, K_in, K_out)
    model_sim = SIS(mode, gamma, d_sim, V, K_in, K_out)

    # All defined moves (includes mixture move)
    moves = subtypes(NetworkPopulations.InvMcmcMove)

    for move in moves

        # Skip mixture move
        if move == NetworkPopulations.InvMcmcMixtureMove
            continue
        end

        move_instance = move() # Move must have default parameters for this to worker

        mcmc_sampler = InvMcmcSampler(
            move_instance,
            desired_samples=10,
            lag=1,
            burn_in=0
        )

        out = mcmc_sampler(model_sis)
        @test out isa McmcOutput

        out = mcmc_sampler(model_sim)
        @test out isa McmcOutput

        # Now we test this move works within a mixture move 
        # Just mix the same move together 50-50
        move_instance_mix = InvMcmcMixtureMove((
                move(), move()
            ), (0.5, 0.5)
        )

        mcmc_sampler_mix = InvMcmcSampler(
            move_instance_mix,
            desired_samples=10,
            lag=1,
            burn_in=0
        )

        out = mcmc_sampler_mix(model_sis)
        @test out isa McmcOutput

        out = mcmc_sampler_mix(model_sim)
        @test out isa McmcOutput

    end


end