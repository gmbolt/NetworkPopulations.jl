# MCMC samplers

* We use iMCMC (reference)
* Workflow 
    1. Construct a move (this can be a mixture of a few)
    2. Construct a sampler with given move
    3. Call sampler on model

## iMCMC sampler 

The sampler for SIS and SIM models is define by instantiating a `InvMcmcSampler` type, which essentially wraps a passed iMCMC move (see section on moves below). 

```@docs 
InvMcmcSampler
```

## iMCMC moves 

We have various different moves, each of which can define a sampler. However, we also have a mixture move which can combine any two moves together. In this way, multiple moves can be flexibly combined in various ways. 

### Mixture moves 

### Defining moves

One can define a new move by sub-typing `InvMcmcMove` and defining the following methods
* 

This new move can then be mixed with others via the `InvMcmcMixturMove` and used within model and posterior samplers.


## Plotting 

* Plot recipes visualise MCMC outputs 
