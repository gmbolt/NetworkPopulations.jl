using RecipesBase, StatsBase, Measures

# # Plot Recipes 
@recipe function f(output::SisMcmcOutput)
    model = output.model
    sample = output.sample
    x = map(x -> model.dist(model.mode, x), sample)
    xguide --> "Sample"
    yguide --> "Distance from Mode"
    legend --> false
    size --> (800, 300)
    margin --> 5mm
    x
end

@recipe function f(output::SimMcmcOutput)
    model = output.model
    sample = output.sample
    x = map(x -> model.dist(model.mode, x), sample)
    xguide --> "Sample"
    yguide --> "Distance from Mode"
    legend --> false
    size --> (800, 300)
    margin --> 5mm
    x
end

@recipe function f(output::SpfMcmcOutput)
    model = output.model
    sample = output.sample
    x = map(x -> model.dist(model.mode, x), sample)
    xguide --> "Sample"
    yguide --> "Distance from Mode"
    legend --> false
    size --> (800, 300)
    margin --> 5mm
    x
end

@userplot LengthPlot

@recipe function f(h::LengthPlot)
    input = h.args
    @assert length(input) == 1 "Accepts only a single argument"
    @assert typeof(input[1]) == SpfMcmcOutput "Input must be of type SpfMcmcOutput"
    size --> (800, 400)
    legend --> false
    data = length.(input[1].sample)
    @series begin
        seriestype := :bar
        yguide --> "Proportion"
        xguide --> "Path Length"
        xticks --> 1:maximum(data)
        xlims --> [0, maximum(data) + 0.5]
        proportionmap(data)
    end

end


