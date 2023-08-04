# Here we define some constructors for backwards compatability with older 
# versions of code. These are mostly due to a change in how the MCMC 
# sampling type hierarchy was constructed.

export make_sampler, SisMcmcInsertDelete, SimMcmcInsertDelete

# Model samplers 
# ---------------
# Used to have separate samplers for each move...

"""
Construct defauly iMCMC, combining path insertion and deletion with 
edit allocation move. 
"""
function make_sampler(
    ; K=100,
    ν_ed=2, ν_td=2,
    β=0.4, # Prob of edit-allocation move
    len_dist=TrGeometric(0.8, 1, K),
    desired_samples=1000, lag=1, burn_in=0, init=InitMode()
)
    move = InvMcmcMixtureMove(
        (
            EditAllocationMove(ν=ν_ed),
            InsertDeleteMove(ν=ν_td, len_dist=len_dist),
        ),
        (β, 1 - β)
    )

    return InvMcmcSampler(
        move,
        desired_samples=desired_samples,
        lag=lag,
        burn_in=burn_in,
        init=init
    )
end

"""
A constructor for a now depereciated sampler class, defined for backwards compatability. 
Will return the equivalent class in the current type hierarchy.

This regards a sampler for the SIS model combining edit allocation move 
with the path insertion deletion move. 
"""
SisMcmcInsertDelete(; kwargs...) = make_sampler(; kwargs...)

"""
A constructor for a now depereciated sampler class, defined for backwards compatability. Will return the equivalent class in the current type hierarchy.


This regards a sampler for the SIM model combining edit allocation move 
with the path insertion deletion move. 
"""
SimMcmcInsertDelete(; kwargs...) = make_sampler(; kwargs...)

