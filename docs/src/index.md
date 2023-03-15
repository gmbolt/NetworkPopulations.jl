```@meta
CurrentModule = NetworkPopulations
```

# Introduction

This is documentation for the [NetworkPopulations.jl](https://github.com/gmbolt/NetworkPopulations.jl) package. 

Some inline latex ``x^2`` 

```math
p(\S| \E^m, \gamma) \propto \exp\{-\gamma d_E(\mathcal{E}, \mathcal{E}^m)\}

```

```@docs
get_normalising_const(::SIS)
```

```julia
function(x)
    print(x)
end     
```

```@example
using Plots, Random # hide
Random.seed!(1) # hide
a = 1
b = 2
a + b
plot(rand(11))
```

!!! note

    The work network can mean different things to different people.