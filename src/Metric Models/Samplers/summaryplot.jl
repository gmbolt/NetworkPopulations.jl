using RecipesBase, Measures, StatsBase


@userplot SummaryPlot
@recipe function f(h::SummaryPlot)
    output = h.args  # A mcmc_output type
    model = output[1].model
    sample = output[1].sample
    layout := (3, 1)
    legend --> false
    xguide := "Sample"
    size --> (800, 600)
    margin --> 5mm

    @series begin
        seriestype := :line
        yguide --> "Dist. from Mode"
        map(x -> model.dist(model.mode, x), sample)
    end
    @series begin
        seriestype := :line
        yguide --> "Num. of Paths"
        map(length, sample)
    end
    @series begin
        seriestype := :line
        yguide --> "Avg. Path Len."
        map(x -> mean(length.(x)), sample)
    end
end