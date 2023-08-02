using NetworkPopulations
using Test

# Note, currently we are making use of NetworkDistances.jl, which is 
# not registered. An initial solution to this is to assume it is accessible 
# from the main Julia environment, i.e. the environment one goes to if 
# Pkg.activate() is called, typicall called @v<version> where version denotes 
# the current julia version. 

# To access this when testing the package we still need to augemnt the LOAD_PATH
# variable, since for some reason it didn't have the necessary entry to 
# permit loading code from this main Julia environment. 
println("Augmenting the LOAD_PATH environment variable...")
println("LOAD_PATH initially: ", LOAD_PATH)
push!(LOAD_PATH, "@v#.#")
println("Updated LOAD_PATH: ", LOAD_PATH)

# Now we should be able to load NetworkDistances
using NetworkDistances

@testset "NetworkPopulations.jl" begin
    include("metric_models/imcmc_moves_test.jl")
end
